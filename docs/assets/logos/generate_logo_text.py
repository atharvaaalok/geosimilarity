import xml.etree.ElementTree as ET

# Load the SVG file
svg_file_path = "logo_text_no_lines.svg"
tree = ET.parse(svg_file_path)
root = tree.getroot()

# Extract the SVG namespace
namespace = {"svg": "http://www.w3.org/2000/svg"}
ET.register_namespace("", namespace["svg"])

# Convert the SVG tree to a string to inspect its structure
svg_string = ET.tostring(root, encoding="unicode")
svg_string[:1000]  # Displaying only the first 1000 characters for inspection


# Extract dimensions of the outer and inner rectangles
outer_rect = root.find(".//svg:rect", namespace)  # First rectangle is the outer one
inner_rects = root.findall(".//svg:rect", namespace)  # Find all rectangles

if len(inner_rects) > 1:
    inner_rect = inner_rects[1]  # Second rectangle is the inner one

    # Extract position and size
    outer_x, outer_y = float(outer_rect.attrib["x"]), float(outer_rect.attrib["y"])
    outer_w, outer_h = float(outer_rect.attrib["width"]), float(outer_rect.attrib["height"])

    inner_x, inner_y = float(inner_rect.attrib["x"]), float(inner_rect.attrib["y"])
    inner_w, inner_h = float(inner_rect.attrib["width"]), float(inner_rect.attrib["height"])

    # Define points on the inner and outer rectangles
    num_lines_left_right = 10
    num_lines_top_bottom = 25
    inner_points = [
        (inner_x + i * inner_w / (num_lines_top_bottom - 1), inner_y) for i in range(num_lines_top_bottom)
    ] + [
        (inner_x + i * inner_w / (num_lines_top_bottom - 1), inner_y + inner_h) for i in range(num_lines_top_bottom)
    ] + [
        (inner_x, inner_y + i * inner_h / (num_lines_left_right - 1)) for i in range(num_lines_left_right)
    ] + [
        (inner_x + inner_w, inner_y + i * inner_h / (num_lines_left_right - 1)) for i in range(num_lines_left_right)
    ]

    outer_points = [
        (outer_x + i * outer_w / (num_lines_top_bottom - 1), outer_y) for i in range(num_lines_top_bottom)
    ] + [
        (outer_x + i * outer_w / (num_lines_top_bottom - 1), outer_y + outer_h) for i in range(num_lines_top_bottom)
    ] + [
        (outer_x, outer_y + i * outer_h / (num_lines_left_right - 1)) for i in range(num_lines_left_right)
    ] + [
        (outer_x + outer_w, outer_y + i * outer_h / (num_lines_left_right - 1)) for i in range(num_lines_left_right)
    ]

    # Add lines connecting inner and outer points
    for inner_pt, outer_pt in zip(inner_points, outer_points):
        line_element = ET.Element("line", {
            "x1": str(inner_pt[0]), "y1": str(inner_pt[1]),
            "x2": str(outer_pt[0]), "y2": str(outer_pt[1]),
            "stroke": "black", "stroke-width": "2"
        })
        root.append(line_element)

    # Save modified SVG
    modified_svg_path = "logo_text.svg"
    tree.write(modified_svg_path)
else:
    modified_svg_path = None