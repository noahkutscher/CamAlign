import cv2
import numpy as np

class ZoomableWindow:
    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image
        self.scale = 1.0
        self.offset = (0, 0)
        self.dragging = False
        self.last_mouse_pos = (0, 0)

        self.scaled_w = image.shape[1]
        self.scaled_h = image.shape[0]

        self.window_size = (1024, 720)

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.additional_callbacks = []
        self.show()

    def add_mouse_callback(self, event, callback):
        self.additional_callbacks.append((event, callback))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.last_mouse_pos = (x, y)
            self.show()
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False                  
            self.show()
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            if self.offset[0] + dx + self.scaled_w > 5 and self.offset[0] + dx < self.window_size[0] - 5:
                self.offset = (self.offset[0] + dx, self.offset[1])
            if self.offset[1] + dy + self.scaled_h > 5 and self.offset[1] + dy < self.window_size[1] - 5:
                self.offset = (self.offset[0], self.offset[1] + dy)
            self.last_mouse_pos = (x, y)
            self.show()
        elif event == cv2.EVENT_MOUSEWHEEL:
            old_scale = self.scale
            if flags > 0:
                self.scale *= 1.1
                shift_x = (x * 1.1) - x
                shift_y = (y * 1.1) - y

            else:
                self.scale /= 1.1
                shift_x = (x / 1.1) - x
                shift_y = (y / 1.1 )- y

            self.scale = max(0.05, min(self.scale, 50.0))

            img_x = (x - self.offset[0]) / old_scale
            img_y = (y - self.offset[1]) / old_scale

            img_x_new = img_x * self.scale + self.offset[0]
            img_y_new = img_y * self.scale + self.offset[1]

            shift_x = img_x_new - x
            shift_y = img_y_new - y

            new_offset = (int(self.offset[0] - shift_x), int(self.offset[1] - shift_y))

            new_off_x = new_offset[0]
            new_off_y = new_offset[1]
            
            if new_off_x + self.scaled_w < 5 or new_off_x > self.window_size[0] - 5:
                self.scale = old_scale     
            elif new_off_y + self.scaled_h < 5 or new_off_y > self.window_size[1] - 5:
                self.scale = old_scale
            else:
                self.offset = new_offset
            
            self.show()

        for cb_event, callback in self.additional_callbacks:
            if event == cb_event:
                callback(x, y, flags, param)

    
    def convert_window_to_image_coords(self, x, y):
        img_x = (x - self.offset[0]) / self.scale
        img_y = (y - self.offset[1]) / self.scale

        if img_x < 0 or img_x >= self.image.shape[1] or img_y < 0 or img_y >= self.image.shape[0]:
            return -1, -1
        
        return int(img_x), int(img_y)

    def show(self):
        h, w = self.image.shape[:2]
        self.scaled_w, self.scaled_h = int(w * self.scale), int(h * self.scale)
        scaled_w, scaled_h = self.scaled_w, self.scaled_h
        
        resized_image = cv2.resize(self.image, (scaled_w, scaled_h))

        display_image = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
        x_offset = self.offset[0]
        y_offset = self.offset[1]

        src_x_start = max(0, -x_offset)
        src_y_start = max(0, -y_offset)

        target_x_offset = max(0, x_offset)
        target_y_offset = max(0, y_offset)

        src_width = min(self.window_size[0] - target_x_offset, scaled_w - src_x_start)
        src_height = min(self.window_size[1] - target_y_offset, scaled_h - src_y_start)

        display_image[target_y_offset:target_y_offset+src_height, target_x_offset:target_x_offset+src_width] = resized_image[src_y_start:src_y_start+src_height, src_x_start:src_x_start+src_width]

        cv2.imshow(self.window_name, display_image)

    def close(self):
        cv2.destroyWindow(self.window_name)

if __name__ == "__main__":
    image = cv2.imread("CamAlign/color.png")
    zoomable_window = ZoomableWindow("Zoomable Window", image)

    while True:
        # zoomable_window.show()
        key = cv2.waitKey(20)
        if key == 27:  # ESC key to exit
            break

    zoomable_window.close()