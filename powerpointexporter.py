from pathlib import Path
from datetime import datetime


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


def _add_picture_fit(slide, image_path, left, top, width, height):
    image_path = str(image_path)
    shape = slide.Shapes.AddPicture(image_path, False, True, 0, 0, -1, -1)
    shape.LockAspectRatio = True

    usable_width = width * 1.04
    usable_height = height * 1.04
    scale = min(usable_width / shape.Width, usable_height / shape.Height)
    shape.Width = shape.Width * scale
    shape.Height = shape.Height * scale
    shape.Left = left + (width - shape.Width) / 2
    shape.Top = top + (height - shape.Height) / 2
    return shape


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

            _replace_slide_pictures(slide)

            for slot_index, image_filename in enumerate(image_filenames):
                image_path = plots_dir / image_filename
                if not image_path.exists():
                    print(f"  Warning: Plot not found for slide {slide_number}: {image_filename}")
                    continue

                if len(picture_boxes) >= len(image_filenames):
                    left, top, width, height = picture_boxes[slot_index]
                else:
                    left, top, width, height = _resolve_box(
                        layout,
                        slide_width,
                        slide_height,
                        slot_index=slot_index,
                        slot_count=len(image_filenames)
                    )
                _add_picture_fit(slide, image_path, left, top, width, height)

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
