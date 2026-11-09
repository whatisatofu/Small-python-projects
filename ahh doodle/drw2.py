import customtkinter as ctk
from functions import *
from PIL import Image
from scipy.ndimage import shift
class DrawPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Drawer")

        # Load trained model
        self.model_W, self.model_b = load_model("mnist_model.pkl")
        # Set up the canvas, like windows?
        self.grid_dimension = 28
        self.cell_size = 20  # Each AI pixel is 20x20 screen pixels
        self.canvas_size = self.grid_dimension * self.cell_size
        
        # Initialize our 28x28 data with zeros (white)
        self.data = np.zeros((28, 28), dtype=np.float32)

        self.canvas = ctk.CTkCanvas(self.root, width=self.canvas_size, 
                                height=self.canvas_size, bg="black", cursor="cross")
        self.canvas.grid(row=1, column=0, padx=20, pady=20)  # Use grid for better layout control
        

        # Add buttons and sliders or so
        self.clear = ctk.CTkButton(self.root, text="Clear", command=self.clear_canvas, corner_radius=10)
        self.clear.grid(row=2, column=0,padx=20, pady=10)

        self.save = ctk.CTkButton(self.root, text="output vector", command=self.output_vector, corner_radius=10)
        self.save.grid(row=3, column=0, padx=20, pady=10)

        # Add text label for listing the guess and confidence
        self.result_label = ctk.CTkLabel(self.root, text="Draw a digit!", font=ctk.CTkFont(size=20))
        self.result_label.grid(row=0, column=0, padx=20, pady=10)

        self.prob_table = ctk.CTkScrollableFrame(self.root, width=200, height=300)
        self.prob_table.grid(row=1, column=1, padx=20, pady=20)

        self.prob_table_label = ctk.CTkLabel(self.prob_table, text="Digit Probabilities", font=ctk.CTkFont(size=16))
        self.prob_table_label.pack(pady=10)

        for i in range(10):
            prob_label = ctk.CTkLabel(self.prob_table, text=f"{i}: 0.0000")
            prob_label.pack(pady=2)

        # Bind mouse events for drawing
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", lambda e: self.predict_digit(self.model_W, self.model_b))  # Predict on mouse release

    def clear_canvas(self):
        self.canvas.delete("all")
        self.data = np.zeros((28, 28), dtype=np.float32)
    
    def output_vector(self):
        self.resized_data = self.resize_and_center(self.data)  # Ensure the data is resized and centered before output
        # Flatten the 28x28 data into a 784-length vector
        flat_vector = np.array(self.resized_data).flatten()
        return flat_vector  # This is the vector you can feed into your model
        
    def paint(self, event):
        # 1. Find which grid cell the mouse is in
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        brush_pattern = [
            (0, 0, 1.0),   # (row_offset, col_offset, strength)
            (1, 0, 0.2), (-1, 0, 0.2), (0, 1, 0.2), (0, -1, 0.2), # Cross neighbors
            (1, 1, 0.05), (-1, -1, 0.05), (1, -1, 0.05), (-1, 1, 0.05) # Diagonal neighbors
        ]

        for dr, dc, weight in brush_pattern:
            r, c = row + dr, col + dc
        # 2. Stay within bounds
            if 0 <= c < 28 and 0 <= r < 28:
                # Update data (1.0 = White)
                new_val = min(1.0, self.data[r][c] + weight)
                self.data[r][c] = new_val
                
                gray_val = int(new_val * 255)
                color_hex = f'#{gray_val:02x}{gray_val:02x}{gray_val:02x}'
                # 3. Draw the block on the canvas
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color_hex, outline="gray")

    def calculate_center(self, image):
        # Calculate the center of mass of the drawn digit
        total_mass = np.sum(image) + 1e-5  # Avoid division by zero
        y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        center_x = np.sum(x * image) / total_mass
        center_y = np.sum(y * image) / total_mass
        return center_x, center_y
    
    def bound_box(self, image):
        # Find the bounding box of the drawn digit
        rows = np.any(image > 0.01, axis=1)
        cols = np.any(image > 0.01, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, image.shape[1], image.shape[0]  # No drawing, return full box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return x_min, y_min, x_max, y_max
    
    def resize_and_center(self, image):
        x_min, y_min, x_max, y_max = self.bound_box(image)
        cropped = image[y_min:y_max+1, x_min:x_max+1]
        
        if cropped.size == 0:
            return np.zeros((28, 28))  # Return blank if nothing drawn
        
        # Resize to fit within 20x20 while maintaining aspect ratio
        h, w = cropped.shape
        scale = min(20 / w, 20 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = np.zeros((28, 28))
        cropped_8bit = (cropped * 255).astype(np.uint8)
        resized_digit = np.array(Image.fromarray(cropped_8bit).resize((new_w, new_h), resample=Image.BILINEAR)) / 255.0
        
        temp_img = np.zeros((28, 28))
        start_y = (28 - new_h) // 2
        start_x = (28 - new_w) // 2
        temp_img[start_y : start_y + new_h, start_x : start_x + new_w] = resized_digit

        cx, cy = self.calculate_center(temp_img)
        shift_x = 13.5 - cx
        shift_y = 13.5 - cy

        # Shift the image to center it
        temp_img = shift(temp_img, shift=(shift_y, shift_x), mode='constant', cval=0.0)
        return temp_img

    def predict_digit(self, W, b):
        # Get the output vector from the canvas
        input_vector = self.output_vector()
        # Reshape to (1, 784) for a single sample
        input_vector = input_vector.reshape(1, -1)
        # Perform forward pass through the model
        activations = forward_pass(input_vector, W, b)
        output_probabilities = activations[-1]  # The last layer's output

        predicted_digit = np.argmax(output_probabilities)
        confidence = np.max(output_probabilities) * 100
        self.result_label.configure(text=f"Predicted: {predicted_digit} with {confidence:.2f}% confidence")
        
        for digit, prob in enumerate(output_probabilities[-1]):
            prob_label = self.prob_table.winfo_children()[digit + 1]  # +1 to skip the title label
            prob_label.configure(text=f"{digit}: {prob:.4f}")

if __name__ == "__main__":
    root = ctk.CTk() 
    app = DrawPanel(root)
    root.state('zoomed')  
    root.mainloop()