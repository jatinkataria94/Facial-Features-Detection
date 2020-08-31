# import the necessary packages
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
from scipy.ndimage.filters import median_filter


'''
Sharpening Filters Information-->

Gaussian blurring is a linear operation. However, it does not preserve edges in the input image - the value of sigma governs the degree of smoothing, and eventually how the edges are preserved.

The Median filter is a non-linear filter. Unlike linear filters, median filters replace the pixel values with the median value available in the local neighborhood (say, 5x5 or 3x3 pixels around the central pixel value). Also, median filter is edge preserving (the median value must actually be the value of one of the pixels in the neighborhood). This is probably a good read: http://arxiv.org/pdf/math/0612422.pdf

Bilateral filter is a non-linear filter. It prevents averaging across image edges, while averaging within smooth regions of the image -> hence edge-preserving. Also, Bilateral filters are non-iterative.
'''


def sharpen_gaussian(image, kernel_size=(5, 5), sigma=1, strength=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(strength + 1) * image - float(strength) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def sharpen_median(image, sigma=5, strength=0.8):

    # Median filtering
    image_mf = median_filter(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf,cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image-strength*lap
    
    return sharp
    
    

def autoFacialFeatures(realtime=False,img_format=None,
                 capture_img=True,image_name=None,
                 face_align=True, radius=3,
                 sharpen=False,sharp_filter='gaussian',
                 n_sharp=None,sigma=1,strength=1.0,
                 sharpen_resize=False,sharpen_resize_dim=(500,500),
                 extract_roi=True,img_width=500):
    
    if realtime==True:
        
        # Load the detector
        detector = dlib.get_frontal_face_detector()
        # Load the predictor
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # read the image
        cap = cv2.VideoCapture(0)
        
        while True:
            _, frame = cap.read()
            # Convert image into grayscale
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
        
            # Use detector to find landmarks
            faces = detector(gray)
        
            for face in faces:
                x1 = face.left()  # left point
                y1 = face.top()  # top point
                x2 = face.right()  # right point
                y2 = face.bottom()  # bottom point
                
                # Draw a rectangle
                cv2.rectangle(img=frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
        
                # Create landmark object
                landmarks = predictor(image=gray, box=face)
        
                # Loop through all the points
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
        
                    # Draw a circle
                    cv2.circle(img=frame, center=(x, y), radius=radius, color=(0, 255, 0), thickness=-1)
        
            # show the image
            cv2.imshow(winname="Face", mat=frame)
        
            # Exit when escape is pressed
            if cv2.waitKey(delay=1) == 27:
                break
        
        # When everything done, release the video capture and video write objects
        cap.release()
        
        # Close all windows
        cv2.destroyAllWindows()
    
    else:
        
        if capture_img==True:
            
            cam = cv2.VideoCapture(0)
        
            cv2.namedWindow("test")
            
            img_counter = 0
            
            while True:
                ret, frame = cam.read()
                if not ret:
                    print("failed to grab frame")
                    break
                cv2.imshow("test", frame)
            
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    print("Escape hit, closing...")
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = ("captured_image_{}."+img_format).format(img_counter)
                    cv2.imwrite(img_name, frame)
                    print("{} written!".format(img_name))
                    img_counter += 1
            
            cam.release()
            
            cv2.destroyAllWindows()
        else:
            img_name=image_name
        
        
        
        
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        if face_align==True:
            fa = FaceAligner(predictor, desiredFaceWidth=img_width)
            
            # load the input image and convert it to grayscale
            image = cv2.imread(img_name)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            rects = detector(grey)
            
            
            
            # loop over the face detections
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
               	# using facial landmarks    
                (x,y,w,h)=rect_to_bb(rect)
                faceOrig=imutils.resize(image[y:y+h, x:x+w], width=img_width)
                faceAligned=fa.align(image, grey, rect)
                #display the output images
                cv2.imshow('Original', faceOrig)
                cv2.imshow('Aligned', faceAligned)
                cv2.waitKey(0)
                
            
            cv2.imwrite(img_name, faceAligned) 
        
        
        if sharpen==True:
            if sharp_filter=='gaussian':
                #call the function to sharpen the image
                image = cv2.imread(img_name)
                if n_sharp==1:
                    sharpened_image = sharpen_gaussian(image, sigma=sigma, strength=strength)
                else:
                    sharpened_image = sharpen_gaussian(image, sigma=sigma, strength=strength)
                    for i in range(1,n_sharp):
                        sharpened_image = sharpen_gaussian(sharpened_image, sigma=sigma, strength=strength)
            
            elif sharp_filter=='median':
                image  = cv2.imread(img_name)
                sharpened_image = np.zeros_like(image)
                if n_sharp==1:
                    for i in range(3):
                        sharpened_image[:,:,i] = sharpen_median(image[:,:,i], sigma=sigma, strength=strength)
                else:
                    for i in range(3):
                        sharpened_image[:,:,i] = sharpen_median(image[:,:,i], sigma=sigma, strength=strength)
                    for i in range(1,n_sharp):
                        for i in range(3):
                            sharpened_image[:,:,i] = sharpen_median(sharpened_image[:,:,i], sigma=sigma, strength=strength)
            
            
            if sharpen_resize==True:
                image=cv2.resize(image, sharpen_resize_dim,interpolation = cv2.INTER_CUBIC) 
                sharpened_image=cv2.resize(sharpened_image, sharpen_resize_dim,interpolation = cv2.INTER_CUBIC) 
                
            cv2.imshow('Original', image)
            cv2.imshow("Sharpened", sharpened_image)
            cv2.imwrite(img_name, sharpened_image)
            cv2.waitKey(0)
            
        
        
        
        image = cv2.imread(img_name)
        #image = cv2.imread(img_name)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        faces = detector(gray)
        
        if extract_roi==True:
            #loop over the face detections
            for (i, face) in enumerate(faces):
                
            	# determine the facial landmarks for the face region, then
            	# convert the landmark (x, y)-coordinates to a NumPy array
            	shape = predictor(image=gray, box=face)        
            	shape = face_utils.shape_to_np(shape)
                
            	# loop over the face parts individually
            	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            		# clone the original image so we can draw on it, then
            		# display the name of the face part on the image
            		clone = image.copy()
            		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            			0.7, (0, 255, 0), 2)
            		# loop over the subset of facial landmarks, drawing the
            		# specific face part
            		for (x, y) in shape[i:j]:
            			cv2.circle(img=clone, center=(x, y), radius=radius, color=(0, 255, 0), thickness=-1)
                    
            		# extract the ROI of the face region as a separate image
            		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            		roi = image[y:y + h, x:x + w]
            		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            		# show the particular face part
            		cv2.imshow("ROI", roi)
            		cv2.imshow("Image", clone)
            		cv2.waitKey(0)
                    
                
        for face in faces:
            x1 = face.left() # left point
            y1 = face.top() # top point
            x2 = face.right() # right point
            y2 = face.bottom() # bottom point
            # Draw a rectangle
            cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=4)
        
           
            
            # Create landmark object
            landmarks = predictor(image=gray, box=face)
            
            # Loop through all the points
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
            
                # Draw a circle
                cv2.circle(img=image, center=(x, y), radius=radius, color=(0, 255, 0), thickness=-1)
        
        
        # show the image
        cv2.imshow(winname="Full Face", mat=image)
        
        # Wait for a key press to exit
        cv2.waitKey(delay=0)
        
        # Close all windows
        cv2.destroyAllWindows()
    

