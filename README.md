# shaking-detection
This python script is used to detect shaking in a video

# ---------------- Usage ----------------
##Running The Program
To run the program, you can run either iteration4.py or manual_test_4.py as any python script.

##Iteration4.py 
Iteration4.py takes the provided video path/rtsp stream/camera stream and analyzes it, printing to the terminal "frame: #; SHAKE" when shaking is detected.

##Manual_test_4.py
Manual_test_4.py automatically applies the same logic as iteration4.py to multiple video files in 2 seperate folders named "shaking" and "notshaking" with files inside named "shaking#.mp4" and "notshaking#.mp4" respectively where # is a number from 1 to the number of files in the folder. Then, it prints to terminal whether or not the provided video is sorted in the right folder, how many frames out of the total were detected as shaking, and at the end, what percent were correct.

##Naming Convention
This script uses snake_case for variable and function names.

##Changing Video Source
To access, different medias, comment or uncomment different CaptureVideo function calls.

##Using RTSP
If you want to try the rtsp connection, make sure you have a corresponding .env file with the required rtsp_url = "rtsp://...".

##Keyboard Interrupts
During the video, you can press r, to force reset the tracked points and q to quit.

##Debug Mode
You can turn on and off debug mode by adjusting the debug boolean. You can also adjust which parts of the debug mode actually work by adjusting the debug variables. For example, the grid drawn represents the grid used to disperse the tracked points. 

##Tuning Variables
Near the top, you will see a large list of variables that can be used to tune the detection code. The comments describe what each one does but in general, the sensitivity variables are going to be the most impactful.


# ---------------- System Summary ----------------
##Base Assumptions
The basis of the system is thus. We assume that shaking requires three things
1. A significant change in the magnitude of acceleration
2. Similar change in velocity across the entire screen
3. An oscillating change in the direction of acceleration over a small period of time

##Assumption Reasoning 
We put these conditions in place because, panning would constitute a change in position, the start and end of a pan would constitute a change in the magnitude and direction of acceleration, and an isolated movement in the fore or background would only cause a limited minority of points to move. 

##General Process
To detect these things, we will track a group of relatively evenly distributed points and calculate position, velocity, acceleration and jerk from that data. 


# ---------------- Preprocessing ----------------
##Why Preprocessing
For the sake of consistency, simplicity, and speed, we process the frames before we analyze any of it as data. 

##Preprocessing Function
Processing the frames mean reducing its resolution and applying a grayscale filter. Although the get_next_frame function actually returns a color and gray frame, both rescaled, one for the sake of viewing and the other for the sake of analyzing.

##Scaling Frames
If their width is greater than the target_width, then the frames are scaled such that the new width is equal to the target_width and the ratio between height and width are maintained.   


# ---------------- Which points do we track? ----------------
##GoodFeaturesToTrack()
The next question is "how do we choose which points we plan on tracking?". OpenCV already provides us with a good start in the form of the goodFeaturesToTrack() function, which uses Shi-tomasi corner detection to find the best points in a given area to track. 

##Weakness of ST-points
However, this itself is not enough since by its nature, Shi-Tomasi points tend to latch onto significant points, which disproportionately include things in the foreground, which is an issue as we do not want to over represent local shaking. 

##Weakness of Even Distribution 
However, picking evenly spread out points also opens a new problem in that the points are often difficult to track and give garbage data, resulting in them being terminated when our data validation occurs, resulting in not enough points to track. 

##Compromise
As a compromise, we split the frame up into a grid of cells and pick a number of Shi_Tomasi points from each cell.


# ---------------- How do we track the points? ----------------
##Tracking Algorithm
As with goodFeaturesToTrack(), OpenCV provides us with a great start point in the calcOpticalFlowPyrLK() function. This function uses the pyramid variation of Lucas-Kanade tracking to calculate the optical flow of a point from one frame to another.

##LK-Tracking Deep Dive
Lucas-Kanade tracking makes 3 assumptions
1. a thing doesnt change that much in between frames as its moving
2. a thing doesn't move to far in between frames
3. pixels generally move together
We basically take a window of pixels, and ask, what translation would cause it to match the most with neighboring pixels in the next frame. However, to do so, we assume that all the pixels in the window have the same movement. Thus, we will have an overdetermined system where we can use least squares to find the best mapping. 

##Pyramid Variation Deep Dive
Although we want to assume that things dont move to fast, it is not always the case. Thus, what we do is we reduce the resolution by a bunch. Then, a 50 pixel movement becomes 5. Then once we get an idea of the mapping, we can slightly increase the resolution and try again but this time with the hint we got from running on a lower resolution. In this way, we can build our way up from a low to a high resolution with greater accuracy. This is the pyramid aspect of the pyramid variation.

##CalcOpticalFlowPyrLK() 
The calcOpticalFlowPyrLK() function returns an array of the translated points it predicted for a set of points given two frames, an array of whether or not a point was found, and an array of the tracking error per point.


# ---------------- Data Validation ----------------
##Need For Data Validation
Sometimes, we lose the points that we are tracking or they are tracked incorrectly. This can lead to garbage or insufficient data that results in inaccurate detection. To safe guard from this, there are two methods we use to prevent this from happening. 

##Minimum Tracked Points
The first is a threshold for the minumum number of points tracked. After we track points to the next frame, if there are less points than the threshold, we reset our points by rerunning the point selection function and wait another 3 frames recollecting data until we are ready to continue detecting. This is controlled by the minpts variable.

##Forward-Backward Tracking Error
The second is a forward backward error validation. The idea is that we track points starting from one frame to the next and once we have the next points, we run a backwards tracking, starting from the next frame with the newly generated points onto the current frame. If the points lie to far away from their original starting position after the roundtrip, they are terminated. This is controlled by the fb_error_threshold variable.

##np.linalg.norm() 
To calculate the magnitude of the error between two points, we use the by default uses the frobenius norm to calculate the general magnitude of a matrix of the two points. Here, we use it to quantify the general magnitude of change when going forwards and backwards once.


# ---------------- Calculations ----------------
##Warm Up = 3
In the first frame (frame 1), we will need to collect our first set of points (stored in good_old). They are stored as their x and y components.

##Warm Up = 2
Then in the second frame (frame 2), we will again collect our next set of tracked points (stored in good_new), to calculate velocity. This is accomplished by calculating the change in position of the x and y component using the equations dx = new_x - old_x and dy = new_y - old_y. We do this for every point we successfully tracked. Assuming one time unit is 1 frame, we can set the change in position equal to the velocity: dx/1 = vx and dy/1 = vy. We then store these velocities in new_vec, which shortly gets shifted into old_vec. Good_new also gets shifted into good_old.

##Warm Up = 1
Next, in the third frame (frame 3), we will again collect our new points, stored into good_new, use these to calculate the next set of velocities (stored into new_vec) Now that we have 2 sets of velocities (new_vec and old_vec), we can calculate the change in velocity in a similar fashion to change in position: dvx = new_vx - old_vx and dvy = new_vy - old_vy1 and thus the acceleration as dvx/1 = ax and dvy/1 = ay. We store the new accelerations into new_accel. Right afterwards, we will take the median of all the accelerations and set that as our global acceleration (stored in global_a). Then we will shift good_new into good_old, new_vec into old_vec, new_accel into old_accel, and global_a into old_global_accel.

##Warm Up = 0
Afterwards, in the fourth frame (frame 4), we will collect our points, calculate our new_vec, new_accel, global_a, and then calculate global jerk (change in acceleration over time) by using the global accelerations (stored in global_j). To do this, we do global_j = da = global_a - old_global_accel. Then, we shift all the new variables to old variables.

##After Warm Up
In the frames afterwards, we will continually recalculate these values for new frames, shifting new values into old values at the end of the loop. Additionally, as soon as we have a valid jerk value (frame 4), we will maintain a queue of a specified size (osc_window) that stores the last couple jerks. 

##Goal 1
Our first goal was to detect a significant change in acceleration. We have done that through our jerk value (global_j) which quantifies the change in acceleration of the median point. If the global_jerk value exceeds a specified threshold (sensitivity), we consider the change in acceleration to be significant.

##Goal 2
Our second goal was to detect a similar change in velocity across the whole screen. We have done that through our global acceleration value (global_a) and our per point acceleration values (new_accel). We can do this per x and y components by reducing our numbers to just their signs (-1, 0, 1) to determine how many individual points are accelerating in the same direction as the median/global value. This will allow us to quantify what ratio of points we track have similar changes in velocity. If our ratio of points exceeds a specified threshold (sensitivity2) then we would consider the change in velocity across the whole screen to be similar. 

##Goal 3
Our third and final goal was to detect an oscillating change in acceleration over a small period of time. We have done that through our dynamic queue of recent jerk values. In a similar fashion to goal two, we can reduce the jerk values to their signs (-1, 0, 1) and see how many times the direction of the jerk changes in the last couple frames. If it changes more than the specified threshold (osc_flips_required),  we would consider the change in acceleration to be oscillating over a small period of time.


# ---------------- Other Notes ----------------
##What Is Jerk?
You'll see a lot of the word "jerk" in the code and this README.md file. But you might be wondering what is jerk? To define jerk, we must first define position, velocity, and acceleration. First, a vector is a value with direction and magnitude. We will be using cartesian coordinates to describe our position with the top left of the video as our anchor point/origin (0,0), increasing as we traverse right and down. One unit of distance will be equal to one pixel and one unit of time will be equal to one frame. Position is a vector describing a location relative to an achor point. Velocity is a vector describing the rate of change of position with respect to time. Acceleration is a vector describing the rate of change of velocity with respect to time. Jerk is a vector describing the rate of change of acceleration with respect to time. In other words, if you took the third-order derivative of a position vs time graph, you would get a graph of jerk vs time. 

##Waitkey()
Curiously, waitkey() is used to advance the frame of the display thus, even if you didn't want keyboard functionality, if you want to see the display you must still keep waitkey. Additionally, since waitkey() adds a delay, messing with it will change the speed of the display. Currently, it is at 1ms (since 0 waits indefinitely) which makes it fine for streams, but much to fast for normal video. 30 ms or even a little less works fine for some videos though the exact wait time is inconsistent as FPS varies between different videos.

##Reset Induced False Negatives
Do note that if you adjust certain variables to the extremes, you will get a large number of false negatives. This is because the program will reset the points once a certain number have stopped being tracked. Additionally, resetting will render the program incapable of recognizing shakes for at least 6 frames (though it can vary). Thus, if you create conditions in which you flood the system with resets, you will not be able to detect any shaking. Turn on resets in debug mode to help debug this issue.

##Minimum Video Length
Theoretically, the shortest video in which you could detect shaking is 4 + osc_flips_required frames. This is because to calculate jerk in the first place, you need 4 different positional data points. This is because jerk is the third-order derivative of position with respect to time. Thus, we would need at least a third order polynomial describing position with respect to time to be able to give a non-zero answer when approximating the jerk. However, we need 4 points to define a third order polynomial, hence the first 4 frames. Afterwards, we need at least osc_flips_required additional frames to add that many extra global jerk values to the jerk history to demonstrate that many flips in direction. However, in practice it would be a good bit longer than that.
