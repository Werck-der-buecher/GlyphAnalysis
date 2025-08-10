import ipywidgets as widgets
from IPython.display import display, clear_output
from ipycanvas import Canvas, MultiCanvas, hold_canvas
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


selected_cmap='viridis'


def get_colormap_dropdown(default: str = 'viridis', description: str = 'Choose colormap:'):
    return widgets.Dropdown(
        options=sorted(plt.colormaps()),
        value=default,
        description=description,
        style={'description_width': 'initial'}
    )


def choose_colormap():
    cmap_dropdown = get_colormap_dropdown()

    # Define a callback function for when the dropdown value changes
    def on_cmap_change(change):
        global selected_cmap
        selected_cmap = change['new']
        print(f"Selected colormap: {selected_cmap}")

    # Attach the callback function to the dropdown widget
    cmap_dropdown.observe(on_cmap_change, names='value')
    return cmap_dropdown


def create_cluster_mapping_widget(group_a_dict, group_b_dict, mapping_dict=None):
    # Extract labels and images from the dictionaries
    group_a_labels = list(group_a_dict.keys())
    group_b_labels = list(group_b_dict.keys())

    # Calculate dynamic canvas height based on the number of clusters
    spacing = 90  # Spacing between boxes
    num_clusters = max(len(group_a_labels), len(group_b_labels))
    canvas_height = max(400, num_clusters * spacing + 60)  # Minimum height of 400, increases as needed

    # Initialize multi-canvas with an overlay canvas for temporary drawings
    canvases = MultiCanvas(2, width=500, height=canvas_height,
                           layout=widgets.Layout(width='500px', height=f'{canvas_height}px'))
    main_canvas, overlay_canvas = canvases[0], canvases[1]
    output = widgets.Output()

    # Initialize the mapping dictionary if none is provided
    if mapping_dict is None:
        mapping_dict = {}

    # Visual settings for boxes
    box_height = 70  # Increased box height for larger images
    box_width = 80
    image_size = (60, 60)  # Larger image size to fit within the box
    radius = 10  # Radius for rounded corners
    start_x_a = 60
    start_x_b = 300
    start_y = 60

    # State variables for tracking connections
    selected_a_index = None  # Index of the selected item in Group A
    temp_line_coords = None  # Temporary line coordinates for visual feedback

    # Resize the images to fit in the box
    def get_resized_image(image_array):
        """Resize and convert an image array to fit within the box as a thumbnail."""
        # Convert float images to uint8 by scaling to [0, 255]
        if image_array.dtype == np.float32 or image_array.dtype == np.float64:
            image_array = ((1-image_array) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(image_array)
        img = img.resize(image_size)
        return np.array(img)

    # Prepare resized images for Group A and Group B
    resized_images_a = {label: get_resized_image(image) for label, image in group_a_dict.items()}
    resized_images_b = {label: get_resized_image(image) for label, image in group_b_dict.items()}

    def draw_static_elements():
        """Draws the boxes and initial connections on the main canvas."""
        with hold_canvas(main_canvas):
            main_canvas.clear()

            # Draw boxes for Group A
            for i, label in enumerate(group_a_labels):
                y = start_y + i * spacing
                image = resized_images_a.get(label)
                draw_rounded_box_with_image(main_canvas, start_x_a, y, box_width, box_height, 'blue',
                                            f"(A) Gr. {label}", image)

            # Draw boxes for Group B
            for i, label in enumerate(group_b_labels):
                y = start_y + i * spacing
                image = resized_images_b.get(label)
                draw_rounded_box_with_image(main_canvas, start_x_b, y, box_width, box_height, 'green',
                                            f"(B) Gr. {label}", image)

            # Draw all confirmed connections based on mapping_dict
            for a_label, b_label in mapping_dict.items():
                if a_label in group_a_labels and b_label in group_b_labels:
                    index_a = group_a_labels.index(a_label)
                    index_b = group_b_labels.index(b_label)
                    y_a = start_y + index_a * spacing + box_height // 2
                    y_b = start_y + index_b * spacing + box_height // 2
                    main_canvas.stroke_style = 'red'
                    main_canvas.line_width = 2
                    main_canvas.stroke_line(start_x_a + box_width, y_a, start_x_b, y_b)

    def draw_rounded_box_with_image(canvas, x, y, width, height, color, label, image=None):
        """Draws a rounded box with an image and a label inside on the specified canvas."""
        canvas.fill_style = color
        canvas.begin_path()
        canvas.move_to(x + radius, y)
        canvas.line_to(x + width - radius, y)
        canvas.arc_to(x + width, y, x + width, y + radius, radius)
        canvas.line_to(x + width, y + height - radius)
        canvas.arc_to(x + width, y + height, x + width - radius, y + height, radius)
        canvas.line_to(x + radius, y + height)
        canvas.arc_to(x, y + height, x, y + height - radius, radius)
        canvas.line_to(x, y + radius)
        canvas.arc_to(x, y, x + radius, y, radius)
        canvas.fill()

        # Draw the label at the top inside the box
        canvas.fill_style = 'white'
        canvas.font = "14px Arial"
        canvas.fill_text(label, x + 5, y + 15)

        # Draw the image inside the box, below the label
        if image is not None:
            image_x, image_y = x + 5, y + 20
            canvas.put_image_data(image, image_x, image_y)

            # Draw a black border around the image
            canvas.stroke_style = 'black'
            canvas.line_width = 1
            canvas.stroke_rect(image_x, image_y, image_size[0], image_size[1])

    # Initial drawing of static elements (boxes and connections)
    draw_static_elements()

    # Helper function to reset and redraw everything on main canvas
    def reset_canvas():
        draw_static_elements()

    # Mouse events for interactive connection drawing
    def on_mouse_down(x, y):
        """Handles the mouse down event to initiate or complete a connection."""
        nonlocal selected_a_index, temp_line_coords

        # Check if clicked inside a Group A box
        for i in range(len(group_a_labels)):
            y_box = start_y + i * spacing
            if start_x_a <= x <= start_x_a + box_width and y_box <= y <= y_box + box_height:
                selected_a_index = i  # Store the selected Group A index
                temp_line_coords = (start_x_a + box_width, y_box + box_height // 2)
                return

        # Check if clicked inside a Group B box and complete the connection
        for j in range(len(group_b_labels)):
            y_box = start_y + j * spacing
            if start_x_b <= x <= start_x_b + box_width and y_box <= y <= y_box + box_height:
                if selected_a_index is not None:
                    # Create mapping
                    a_label = group_a_labels[selected_a_index]
                    b_label = group_b_labels[j]

                    # Remove any previous connection to this Group B item
                    for key, value in list(mapping_dict.items()):
                        if value == b_label:
                            del mapping_dict[key]

                    # Add the new connection
                    mapping_dict[a_label] = b_label
                    selected_a_index = None
                    temp_line_coords = None  # Clear temporary line
                    reset_canvas()
                    with output:
                        clear_output(wait=True)
                        print("Current Mappings:")
                        for key, value in mapping_dict.items():
                            print(f"{key} -> {value}")
                return

    def on_mouse_move(x, y):
        """Handles the mouse move event to show a temporary line for feedback on overlay canvas."""
        nonlocal temp_line_coords

        overlay_canvas.clear()  # Only clear the overlay canvas

        # Only draw the temporary line if an item in Group A is selected
        if selected_a_index is not None:
            with hold_canvas(overlay_canvas):
                overlay_canvas.stroke_style = 'gray'
                overlay_canvas.line_width = 1
                overlay_canvas.stroke_line(temp_line_coords[0], temp_line_coords[1], x, y)

    # Attach mouse events to overlay canvas
    overlay_canvas.on_mouse_down(on_mouse_down)
    overlay_canvas.on_mouse_move(on_mouse_move)

    # Button to clear all connections
    def on_clear_connections_button_clicked(b):
        mapping_dict.clear()  # Clear the mapping dictionary
        reset_canvas()  # Redraw boxes without connections
        overlay_canvas.clear()  # Clear any temporary lines
        with output:
            clear_output(wait=True)
            print("All connections have been cleared.")

    clear_connections_button = widgets.Button(description="Clear All Connections")
    clear_connections_button.on_click(on_clear_connections_button_clicked)

    # Dropdowns and button for manual mapping
    dropdown_a = widgets.Dropdown(options=group_a_labels, description="Group A:")
    dropdown_b = widgets.Dropdown(options=group_b_labels, description="Group B:")
    add_mapping_button = widgets.Button(description="Create Mapping")

    def on_add_mapping_button_clicked(b):
        """Handles manual mapping via dropdowns."""
        a_label = dropdown_a.value
        b_label = dropdown_b.value

        # Remove any previous connection to this Group B item
        for key, value in list(mapping_dict.items()):
            if value == b_label:
                del mapping_dict[key]

        # Add the new connection
        mapping_dict[a_label] = b_label
        reset_canvas()  # Redraw to show updated mappings
        overlay_canvas.clear()  # Clear any temporary lines
        with output:
            clear_output(wait=True)
            print("Current Mappings:")
            for key, value in mapping_dict.items():
                print(f"{key} -> {value}")

    add_mapping_button.on_click(on_add_mapping_button_clicked)

    # Display widgets
    display(
        widgets.VBox([
            widgets.HBox([dropdown_a, dropdown_b, add_mapping_button]),
            clear_connections_button,
            canvases,  # Display both main and overlay canvases
            output
        ])
    )

    # Return the mapping dictionary for access after widget use
    return mapping_dict
