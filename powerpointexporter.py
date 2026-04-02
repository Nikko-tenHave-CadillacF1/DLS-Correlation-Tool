from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET


MAIN_PLOT_BOX = {
    'left_ratio': 0.079,
    'top_ratio': 0.260,
    'width_ratio': 0.90,
    'height_ratio': 0.65,
}

DOUBLE_PLOT_LAYOUT = {
    'left_ratio': 0.0,
    'top_ratio': 0.245,
    'width_ratio': 1.2,
    'height_ratio': 0.9,
    'gap_ratio': 0.0,
}

MSO_PICTURE_TYPES = {11, 13}
PPTX_NAMESPACES = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
}


def _resolve_box(layout_name, slide_width, slide_height, slot_index=0, slot_count=1):
    if layout_name == 'main_plot':
        box = MAIN_PLOT_BOX
        return (
            slide_width * box['left_ratio'],
            slide_height * box['top_ratio'],
            slide_width * box['width_ratio'],
            slide_height * box['height_ratio'],
        )

    if layout_name == 'double_plot':
        layout = DOUBLE_PLOT_LAYOUT
        total_left = slide_width * layout['left_ratio']
        total_top = slide_height * layout['top_ratio']
        total_width = slide_width * layout['width_ratio']
        total_height = slide_height * layout['height_ratio']
        gap = slide_width * layout['gap_ratio']
        slot_width = (total_width - gap) / 2
        left = total_left + slot_index * (slot_width + gap)
        return left, total_top, slot_width, total_height

    raise ValueError(f"Unsupported PowerPoint layout: {layout_name}")


def _replace_slide_pictures(slide):
    for shape_index in range(slide.Shapes.Count, 0, -1):
        shape = slide.Shapes(shape_index)
        if shape.Type in MSO_PICTURE_TYPES:
            shape.Delete()


def _get_picture_boxes(slide):
    picture_boxes = []
    for shape_index in range(1, slide.Shapes.Count + 1):
        shape = slide.Shapes(shape_index)
        if shape.Type in MSO_PICTURE_TYPES:
            picture_boxes.append((shape.Left, shape.Top, shape.Width, shape.Height))
    picture_boxes.sort(key=lambda box: (box[0], box[1]))
    return picture_boxes


def _get_double_plot_boxes(picture_boxes, slide_width, slide_height):
    """Expand the two-up layout within the template's overall double-plot area."""
    if len(picture_boxes) >= 2:
        left = min(box[0] for box in picture_boxes)
        top = min(box[1] for box in picture_boxes)
        right = max(box[0] + box[2] for box in picture_boxes)
        bottom = max(box[1] + box[3] for box in picture_boxes)

        sorted_boxes = sorted(picture_boxes, key=lambda box: box[0])
        detected_gap = sorted_boxes[1][0] - (sorted_boxes[0][0] + sorted_boxes[0][2])
        gap = max(detected_gap, slide_width * 0.01)
    else:
        layout = DOUBLE_PLOT_LAYOUT
        left = slide_width * layout['left_ratio']
        top = slide_height * layout['top_ratio']
        total_width = slide_width * layout['width_ratio']
        total_height = slide_height * layout['height_ratio']
        gap = slide_width * layout['gap_ratio']
        right = left + total_width
        bottom = top + total_height

    total_width = right - left
    total_height = bottom - top
    slot_width = max((total_width - gap) / 2, 0)

    return [
        (left, top, slot_width, total_height),
        (left + slot_width + gap, top, slot_width, total_height),
    ]


def _get_main_plot_box(picture_boxes, slide_width, slide_height):
    """Expand the single main plot to use the full slide width."""
    if picture_boxes:
        _, top, _, height = picture_boxes[0]
        return (0, top, slide_width, height)

    left, top, width, height = _resolve_box('main_plot', slide_width, slide_height)
    return (0, top, slide_width, height)


def _add_picture_fit(slide, image_path, left, top, width, height, fill_factor=1.0):
    image_path = str(image_path)
    shape = slide.Shapes.AddPicture(image_path, False, True, 0, 0, -1, -1)
    shape.LockAspectRatio = True

    scale = min(width / shape.Width, height / shape.Height)
    scale *= fill_factor
    shape.Width = shape.Width * scale
    shape.Height = shape.Height * scale
    shape.Left = left + (width - shape.Width) / 2
    shape.Top = top + (height - shape.Height) / 2
    shape.Line.Visible = True
    shape.Line.ForeColor.RGB = 0
    shape.Line.Weight = 1
    return shape


def get_template_plot_aspect_ratios(template_path, export_map):
    """Return target aspect ratios for exported plots using template picture boxes."""
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template not found: {template_path}")

    aspect_ratios = {}
    with ZipFile(template_path) as pptx_file:
        presentation_root = ET.fromstring(pptx_file.read("ppt/presentation.xml"))
        slide_size = presentation_root.find("p:sldSz", PPTX_NAMESPACES)
        slide_width = int(slide_size.attrib["cx"]) if slide_size is not None else None

        for slide_number, slide_config in export_map.items():
            slide_xml = f"ppt/slides/slide{slide_number}.xml"
            if slide_xml not in pptx_file.namelist():
                continue

            root = ET.fromstring(pptx_file.read(slide_xml))
            picture_boxes = []
            for picture in root.findall('.//p:pic', PPTX_NAMESPACES):
                transform = picture.find('p:spPr/a:xfrm', PPTX_NAMESPACES)
                if transform is None:
                    continue
                ext = transform.find('a:ext', PPTX_NAMESPACES)
                if ext is None:
                    continue
                width = int(ext.attrib['cx'])
                height = int(ext.attrib['cy'])
                if width > 0 and height > 0:
                    offset = transform.find('a:off', PPTX_NAMESPACES)
                    left = int(offset.attrib['x']) if offset is not None else 0
                    top = int(offset.attrib['y']) if offset is not None else 0
                    picture_boxes.append((left, top, width, height))

            picture_boxes.sort(key=lambda box: (box[0], box[1]))
            image_filenames = slide_config.get('images', [])
            slide_aspects = []
            for slot_index, image_filename in enumerate(image_filenames):
                if slot_index >= len(picture_boxes):
                    break
                _, _, width, height = picture_boxes[slot_index]
                if (
                    slide_config.get('layout') == 'main_plot'
                    and slide_width is not None
                    and len(image_filenames) == 1
                ):
                    width = slide_width
                slide_aspects.append((image_filename, width / height))

            if (
                slide_config.get('layout') == 'double_plot'
                and len(slide_aspects) == 2
                and not all(image_filename.startswith('scatter_') for image_filename, _ in slide_aspects)
            ):
                average_aspect = sum(aspect for _, aspect in slide_aspects) / len(slide_aspects)
                for image_filename, _ in slide_aspects:
                    aspect_ratios[image_filename] = (average_aspect,)
            else:
                for image_filename, aspect_ratio in slide_aspects:
                    aspect_ratios[image_filename] = aspect_ratio

    return aspect_ratios


def export_report_to_powerpoint(template_path, output_path, plots_dir, export_map, visible=False):
    try:
        import win32com.client
    except ImportError as exc:
        raise ImportError(
            "pywin32 is required for PowerPoint export. Install requirements with "
            "'pip install -r requirements.txt'."
        ) from exc

    template_path = Path(template_path).resolve()
    output_path = Path(output_path).resolve()
    plots_dir = Path(plots_dir).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template not found: {template_path}")

    powerpoint = win32com.client.Dispatch("PowerPoint.Application")
    # PowerPoint does not allow hiding the application via Application.Visible = False.
    powerpoint.Visible = True
    presentation = None

    try:
        presentation = powerpoint.Presentations.Open(str(template_path), WithWindow=visible)
        slide_width = presentation.PageSetup.SlideWidth
        slide_height = presentation.PageSetup.SlideHeight

        for slide_number, slide_config in export_map.items():
            slide = presentation.Slides(slide_number)
            layout = slide_config['layout']
            image_filenames = slide_config['images']
            picture_boxes = _get_picture_boxes(slide)

            if layout == 'main_plot' and len(image_filenames) == 1:
                target_boxes = [_get_main_plot_box(picture_boxes, slide_width, slide_height)]
            elif layout == 'double_plot' and len(image_filenames) == 2:
                target_boxes = _get_double_plot_boxes(picture_boxes, slide_width, slide_height)
            else:
                target_boxes = picture_boxes

            _replace_slide_pictures(slide)

            for slot_index, image_filename in enumerate(image_filenames):
                image_path = plots_dir / image_filename
                if not image_path.exists():
                    print(f"  Warning: Plot not found for slide {slide_number}: {image_filename}")
                    continue

                if len(target_boxes) >= len(image_filenames):
                    left, top, width, height = target_boxes[slot_index]
                else:
                    left, top, width, height = _resolve_box(
                        layout,
                        slide_width,
                        slide_height,
                        slot_index=slot_index,
                        slot_count=len(image_filenames)
                    )
                if layout == 'double_plot' and image_filename.startswith('scatter_'):
                    fill_factor = 1.15
                elif layout == 'double_plot' and image_filename.startswith('psd_'):
                    fill_factor = 1.15
                else:
                    fill_factor = 1.0
                _add_picture_fit(slide, image_path, left, top, width, height, fill_factor=fill_factor)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            presentation.SaveAs(str(output_path))
            saved_path = output_path
        except Exception as exc:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            print(
                f"  Warning: Could not save PowerPoint report to {output_path} "
                f"({exc}). Saving to {fallback_path} instead."
            )
            presentation.SaveAs(str(fallback_path))
            saved_path = fallback_path

        print(f"PowerPoint report saved to: {saved_path}")
    finally:
        if presentation is not None:
            presentation.Close()
        powerpoint.Quit()
