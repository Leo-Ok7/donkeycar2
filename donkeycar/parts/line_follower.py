import cv2
import numpy as np
from simple_pid import PID
import logging

logger = logging.getLogger(__name__)


class LineFollower:
    '''
    OpenCV based controller
    This controller takes a horizontal slice of the image at a set Y coordinate.
    Then it converts to HSV and does a color thresh hold to find the yellow pixels.
    It does a histogram to find the pixel of maximum yellow. Then is uses that iPxel
    to guid a PID controller which seeks to maintain the max yellow at the same point
    in the image.
    '''
    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y   # num pixels from the top to start horiz scan
        self.scan_height = cfg.SCAN_HEIGHT  # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)  # hsv dark yellow
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)  # hsv light yellow
        self.target_pixel = cfg.TARGET_PIXEL  # of the N slots above, which is the ideal relationship target
        # pixels ignored down each side of the scan band; 0 = use full width
        self.scan_x_margin = int(getattr(cfg, 'SCAN_X_MARGIN', 0))
        # Hold the LATERAL steering response constant as speed changes; see
        # run(). Baseline is the throttle the PID gains were tuned at.
        self.speed_scaled_steering = bool(
            getattr(cfg, 'STEERING_SPEED_SCALING', True))
        self.steering_baseline_throttle = float(
            getattr(cfg, 'STEERING_BASELINE_THROTTLE', cfg.THROTTLE_INITIAL))
        # Exponential smoothing of the measured line position. 1.0 = raw.
        self.line_smoothing = float(getattr(cfg, 'LINE_SMOOTHING_ALPHA', 1.0))
        self.smoothed_x = None
        # Behaviour when the line is not visible; see run().
        self.lost_hold_frames = int(getattr(cfg, 'LINE_LOST_HOLD_FRAMES', 10))
        self.lost_decay = float(getattr(cfg, 'LINE_LOST_DECAY', 0.90))
        self.lost_count = 0
        # Progressive steering; see run(). 0.0 = plain linear response.
        self.progressive = float(getattr(cfg, 'STEERING_PROGRESSIVE', 0.0))
        # Minimum connected-component size, in pixels; <=1 disables.
        self.min_blob_area = int(getattr(cfg, 'LINE_MIN_BLOB_AREA', 0))
        self.target_threshold = cfg.TARGET_THRESHOLD # minimum distance from target_pixel before a steering change is made.
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD  # percentage of yellow pixels that must be in target_pixel slice
        self.steering = 0.0 # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP  # how much to change throttle when off
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.pid_st = pid


    def get_i_color(self, cam_img):
        '''
        get the horizontal position of the color at the given slice of the image
        input: cam_image, an RGB numpy array
        output: (centroid_x, confidence, mask)
            centroid_x - intensity-weighted mean x of the yellow pixels, as a
                         float, or None when no yellow was found at all.
            confidence - fraction of the sampled pixels that are yellow, in
                         0.0 .. 1.0.
            mask       - the binary mask of in-range pixels.

        Position is the CENTROID rather than argmax. argmax returns whichever
        single column happens to hold the most yellow, so it snaps between
        columns on a line that has real width, and a couple of stray pixels
        anywhere in the slice can win outright. Averaging over every yellow
        pixel uses the whole line and moves smoothly as it drifts.

        Note on confidence: this is now a FRACTION of sampled pixels, which is
        what cfg.CONFIDENCE_THRESHOLD is documented to be ("the fraction of
        total sampled pixels that must be yellow"). Previously this returned
        hist[max_yellow] -- a sum in which every yellow pixel contributes 255 --
        so a single yellow pixel produced 255, which cleared the 0.0015
        threshold by five orders of magnitude. The check could never fail, and
        "No line detected" never fired no matter how little yellow was present.
        '''
        # take a horizontal slice of the image
        iSlice = self.scan_y
        scan_line = cam_img[iSlice : iSlice + self.scan_height, :, :]

        # convert to HSV color space
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # make a mask of the colors in our range we are looking for
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # Ignore a margin down each side. The gravel bed, kerb and grass sit at
        # the edges of the frame and share the tape's hue closely enough that
        # no colour threshold separates them; the line being followed stays
        # near the middle. Blanking the margin in the mask (rather than
        # slicing the array) keeps every column index in full-frame
        # coordinates, so the centroid can still be compared directly against
        # target_pixel without an offset.
        margin = self.scan_x_margin
        if margin > 0:
            width = mask.shape[1]
            margin = min(margin, max(0, (width // 2) - 1))
            mask[:, :margin] = 0
            mask[:, width - margin:] = 0
            searched = mask.shape[0] * max(1, width - 2 * margin)
        else:
            searched = mask.size

        # Drop tiny disconnected specks before measuring position.
        #
        # The saturation floor has to be low to catch WORN tape, and that also
        # lets isolated pavement grains through. They are individually tiny but
        # numerous -- measured on real frames, up to 406 components in a single
        # band, where the tape itself is one blob of 150-700px. Because the
        # position is the centroid of every yellow pixel, each speck tugs the
        # result, and specks scattered to one side bias the steering.
        #
        # Real tape is CONNECTED and substantial; noise is not. Removing
        # components below min_blob_area keeps the tape untouched while
        # discarding the confetti.
        if self.min_blob_area > 1:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            if count > 1:
                keep = np.zeros(count, dtype=bool)
                keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= self.min_blob_area
                mask = np.where(keep[labels], mask, 0).astype(mask.dtype)

        # per-column yellow weight, then the weighted mean column index
        hist = np.sum(mask, axis=0, dtype=np.float64)
        total_weight = hist.sum()

        # fraction of the SEARCHED pixels that are yellow, so the threshold
        # keeps its meaning when a margin is cropped off
        confidence = float(np.count_nonzero(mask)) / float(searched)

        if total_weight <= 0.0:
            # nothing yellow in the slice; let the caller hold its heading
            return None, confidence, mask

        columns = np.arange(mask.shape[1], dtype=np.float64)
        centroid = float((columns * hist).sum() / total_weight)

        return centroid, confidence, mask


    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the image.
        If overlay_image is True, then the output image
        includes and overlay that shows how the 
        algorithm is working; otherwise the image
        is just passed-through untouched. 
        '''
        if cam_img is None:
            # three outputs, matching the normal return below -- this used to
            # return four, which raises as soon as the camera hands over a
            # None frame.
            return 0, 0, cam_img

        max_yellow, confidence, mask = self.get_i_color(cam_img)

        # Smooth the measured position before steering on it.
        #
        # The centre line is DASHED, and the scan band is tall enough to hold
        # more than one dash. As the car drives, dashes enter and leave the
        # band, so the centroid of "all yellow in the band" hops even when the
        # car is travelling perfectly straight. Measured on real consecutive
        # frames: median frame-to-frame jump 4.8px, but p90 25px and worst
        # 146px. A car cannot translate 146px sideways in one 50ms frame, so
        # those large steps are measurement artefacts, not real error -- and
        # each one drove a full steering correction.
        #
        # An exponential moving average keeps genuine drift (which persists
        # across frames and accumulates) while suppressing single-frame hops
        # from a dash appearing or disappearing.
        if max_yellow is not None and confidence >= self.confidence_threshold:
            if self.smoothed_x is None or self.line_smoothing >= 1.0:
                self.smoothed_x = max_yellow
            else:
                a = self.line_smoothing
                self.smoothed_x = a * max_yellow + (1.0 - a) * self.smoothed_x
            max_yellow = self.smoothed_x
        elif max_yellow is None:
            # line genuinely gone -- drop the filter so it re-acquires cleanly
            # rather than easing over from a stale position
            self.smoothed_x = None

        if self.target_pixel is None:
            # Centre of the image, so "centred" means centred in the camera
            # frame. This used to latch onto wherever the line happened to be
            # on the very first frame, which made the whole notion of centred
            # depend on how the car was sitting at startup -- and if that first
            # frame caught noise instead of the line, every later correction was
            # measured against the wrong reference for the rest of the run.
            # Set cfg.TARGET_PIXEL explicitly to deliberately ride offset.
            self.target_pixel = cam_img.shape[1] // 2
            logger.info(f"Target line position defaulted to image centre = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            # this is the target of our steering PID controller
            self.pid_st.setpoint = self.target_pixel

        if max_yellow is not None and confidence >= self.confidence_threshold:
            # Deadband: once the line is within target_threshold pixels of the
            # target there is nothing worth correcting, and steering on that
            # residual error just makes the car saw back and forth around
            # centre. Feeding the setpoint itself -- rather than skipping the
            # call -- keeps the PID running on zero error, so its internal
            # state stays current and any accumulated term unwinds instead of
            # being frozen at whatever it held when the car went straight.
            error = max_yellow - self.target_pixel
            if abs(error) <= self.target_threshold:
                steering = self.pid_st(self.target_pixel)
            else:
                # Progressive response. One linear gain cannot serve both jobs
                # here: measured over 1500 real frames the line sits a median
                # 78px off centre and 40% of frames exceed 80px, so a gain low
                # enough to stop weaving on the straight (Kp=-0.005) tops out
                # at 0.70 steering even at the largest error seen (148px) --
                # not enough to get round a corner, which is exactly where it
                # was losing the line.
                #
                # Stretching the error by how far off it already is keeps the
                # response near-linear for small deviations (so the straight
                # stays calm) while letting a corner-sized error reach full
                # lock. With progressive=1.0 a 124px error acts like ~196px,
                # i.e. 0.62 -> 0.98 steering, while a 40px error moves only
                # 0.20 -> 0.24.
                shaped = max_yellow
                if self.progressive > 0.0:
                    half = cam_img.shape[1] / 2.0
                    stretch = 1.0 + self.progressive * min(1.0, abs(error) / half)
                    shaped = self.target_pixel + error * stretch
                steering = self.pid_st(shaped)

            # Speed-scaled steering.
            #
            # A steering command is geometric, but how far it actually moves
            # the car sideways is (steering x speed x time). So a gain that is
            # calm at THROTTLE_INITIAL is roughly twice as aggressive at twice
            # the throttle -- the car over-corrects, overshoots, and weaves.
            # Dividing by the throttle ratio holds the LATERAL response
            # constant, so one set of PID gains works across the speed range
            # instead of only at the speed it was tuned at.
            #
            # Clamped at 1.0 so this only ever softens steering when going
            # faster than baseline; it never amplifies at low speed, where a
            # bigger correction would be the wrong thing near a stop.
            if self.speed_scaled_steering and self.throttle > 0.01:
                scale = self.steering_baseline_throttle / self.throttle
                steering *= min(1.0, max(0.25, scale))

            # Clamp to the documented output range. The PID is constructed
            # without output_limits, so with Kp=-0.01 a 200px error already
            # yields 2.0 and the D term can spike far past that -- measured on
            # real tub frames, 88% of commands exceeded 1.0, peaking at 15.95.
            # Anything outside -1..1 is meaningless to the drivetrain: the VESC
            # maps angle to servo as (angle * 0.5) + 0.5, so 15.95 asks for
            # servo position 8.5 on a 0..1 output. That is full lock every
            # frame, which is indistinguishable from a hard left/right slam.
            self.steering = max(-1.0, min(1.0, steering))
            self.lost_count = 0

            # slow down linearly when away from ideal, and speed up when close
            if abs(max_yellow - self.target_pixel) > self.target_threshold:
                # we will be turning, so slow down
                if self.throttle > self.throttle_min:
                    self.throttle -= self.delta_th
                if self.throttle < self.throttle_min:
                    self.throttle = self.throttle_min
            else:
                # we are going straight, so speed up
                if self.throttle < self.throttle_max:
                    self.throttle += self.delta_th
                if self.throttle > self.throttle_max:
                    self.throttle = self.throttle_max
        else:
            # Line not visible this frame.
            #
            # Holding the last steering forever is what drives the car off the
            # track: the centre line is dashed and some stretches have no tape
            # at all, and measured on real recorded frames these gaps run to
            # 5.7 SECONDS (113 frames at 20Hz), median 0.58s. Holding a turn
            # for five seconds is a guaranteed departure.
            #
            # So: hold briefly, which is right for an ordinary dash gap and
            # keeps the car committed through it, then ease the steering back
            # toward straight. Straight is the safe default when blind -- it
            # keeps whatever heading the car already had instead of continuing
            # to curve away from the track.
            self.lost_count += 1
            if self.lost_count > self.lost_hold_frames:
                self.steering *= self.lost_decay
                if abs(self.steering) < 0.01:
                    self.steering = 0.0
            if max_yellow is None:
                logger.info("No line detected: no yellow pixels in the scan slice")
            else:
                logger.info(f"No line detected: confidence {confidence:.5f} < {self.confidence_threshold}")

        # show some diagnostics
        if self.overlay_image:
            # overlay_display formats this with "{:d}", so hand it an int --
            # the centroid is a float now, and is None when nothing was found.
            i_display = int(round(max_yellow)) if max_yellow is not None else -1
            cam_img = self.overlay_display(cam_img, mask, i_display, confidence)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, max_yellow, confidense):
        '''
        composite mask on top the original image.
        show some values we are using for control
        '''

        iSlice = self.scan_y
        img = np.copy(cam_img)
        band = img[iSlice : iSlice + self.scan_height, :, :]

        # TINT the matched pixels rather than replacing the band with the raw
        # 0/255 mask. The old version overwrote those rows entirely, producing
        # a hard black stripe with white speckles -- which hid the very thing
        # you need to see, namely WHAT the mask landed on. Blending keeps the
        # scene visible underneath, so it is obvious at a glance whether the
        # match is the line or the pavement/gravel next to it.
        matched = mask > 0
        if np.any(matched):
            tinted = band.copy()
            tinted[matched] = (255, 0, 255)          # magenta = matched
            img[iSlice : iSlice + self.scan_height, :, :] = cv2.addWeighted(
                band, 0.45, tinted, 0.55, 0)

        # Edges of the scan slice, so the sampled rows are unambiguous.
        cv2.line(img, (0, iSlice), (img.shape[1], iSlice), (0, 255, 255), 1)
        cv2.line(img, (0, iSlice + self.scan_height),
                 (img.shape[1], iSlice + self.scan_height), (0, 255, 255), 1)

        # Where we want the line (white) vs where it actually is (green).
        # Their separation IS the steering error, so the two together show at a
        # glance whether the controller is chasing the right thing.
        if self.target_pixel is not None:
            tx = int(self.target_pixel)
            cv2.line(img, (tx, iSlice - 6),
                     (tx, iSlice + self.scan_height + 6), (255, 255, 255), 1)
        if max_yellow is not None and max_yellow >= 0:
            cx = int(max_yellow)
            cv2.line(img, (cx, iSlice - 6),
                     (cx, iSlice + self.scan_height + 6), (0, 255, 0), 2)

        display_str = []
        display_str.append("STEERING:{:.2f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        if max_yellow is not None and max_yellow >= 0:
            display_str.append("LINE X:{:d}  TARGET:{}".format(
                int(max_yellow), self.target_pixel))
        else:
            display_str.append("LINE X: none  TARGET:{}".format(self.target_pixel))
        # confidence is a fraction of sampled pixels; show it as a percentage
        # with enough digits to compare against CONFIDENCE_THRESHOLD.
        display_str.append("CONF:{:.3f}%  (min {:.3f}%)".format(
            confidense * 100.0, self.confidence_threshold * 100.0))

        y = 12
        x = 8
        for s in display_str:
            # black text with a light halo, so it stays readable over both the
            # bright pavement and the dark shadow the original text vanished in
            cv2.putText(img, s, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=(255, 255, 255), thickness=2)
            cv2.putText(img, s, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4, color=(0, 0, 0), thickness=1)
            y += 12

        return img

