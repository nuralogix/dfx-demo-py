import cv2
background = cv2.imread("ui/background.png")
frame = cv2.imread("ui/frame.png")
innercircle = cv2.imread("ui/inner_circle.png")
intro = cv2.imread("ui/intro.png",cv2.IMREAD_UNCHANGED)

def ui_loop():
    window_name = "Display"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


    while True:
        # read state without blocking event loop
        
      #  cv2.imshow(window_name, background)
        cv2.imshow(window_name, frame)
      #  cv2.imshow(window_name, intro)
        

        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    ui_loop()