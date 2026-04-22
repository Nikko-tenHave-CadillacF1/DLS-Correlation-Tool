"""PowerPoint export helpers."""

from pathlib import Path
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET


# ================================================================
# CONSTANT LAYOUT DEFINITIONS
# ================================================================

MAIN_PLOT_BOX = {
    "left_ratio": 0.079,
    "top_ratio": 0.260,
    "width_ratio": 0.90,
    "height_ratio": 0.65,
}

DOUBLE_PLOT_LAYOUT = {
    "left_ratio": 0.0,
    "top_ratio": 0.245,
    "width_ratio": 1.2,
    "height_ratio": 0.9,
    "gap_ratio": 0.0,
}

# PowerPoint picture MsoShapeType values
MSO_PICTURE_TYPES = {11, 13}

PPTX_NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


# ================================================================
# GEOMETRY HELPERS
# ================================================================

def _resolve_box(layout_name, slide_width, slide_height, slot_index=0, slot_count=1):
    """Compute a plot box for main/double layouts."""
    if layout_name == "main_plot":
        box = MAIN_PLOT_BOX
        return (
            slide_width * box["left_ratio"],
            slide_height * box["top_ratio"],
            slide_width * box["width_ratio"],
            slide_height * box["height_ratio"],
        )

    if layout_name == "double_plot":
        box = DOUBLE_PLOT_LAYOUT
        left = slide_width * box["left_ratio"]
        top = slide_height * box["top_ratio"]
        width = slide_width * box["width_ratio"]
        height = slide_height * box["height_ratio"]
        gap = slide_width * box["gap_ratio"]

        slot_width = (width - gap) / 2
        return (
            left + slot_index * (slot_width + gap),
            top,
            slot_width,
            height,
        )

    raise ValueError(f"Unsupported PowerPoint layout: {layout_name}")


# ================================================================
# SLIDE TEMPLATE PARSING
# ================================================================

def _replace_slide_pictures(slide):
    """Delete all picture shapes from a slide."""
    # reversed loop because PowerPoint collection mutates on delete
    for idx in range(slide.Shapes.Count, 0, -1):
        shape = slide.Shapes(idx)
        if shape.Type in MSO_PICTURE_TYPES:
            shape.Delete()


def _get_picture_boxes(slide):
    """Extract picture bounding boxes from a slide."""
    boxes = []
    for idx in range(1, slide.Shapes.Count + 1):
        sh = slide.Shapes(idx)
        if sh.Type in MSO_PICTURE_TYPES:
            boxes.append((sh.Left, sh.Top, sh.Width, sh.Height))

    return sorted(boxes, key=lambda b: (b[0], b[1]))


def _get_double_plot_boxes(picture_boxes, slide_width, slide_height):
    """
    For double-plot layouts:
    - If template already contains picture placeholders, use those.
    - Otherwise derive new ones using the DOUBLE_PLOT_LAYOUT ratios.
    """
    if len(picture_boxes) >= 2:
        # Use template-detected bounding boxes
        sorted_boxes = sorted(picture_boxes, key=lambda b: b[0])
        return sorted_boxes[:2]

    # Fall back to generic
    layout = DOUBLE_PLOT_LAYOUT
    left = slide_width * layout["left_ratio"]
    top = slide_height * layout["top_ratio"]
    total_width = slide_width * layout["width_ratio"]
    total_height = slide_height * layout["height_ratio"]
    gap = slide_width * layout["gap_ratio"]

    slot_width = max((total_width - gap) / 2, 0)

    return [
        (left, top, slot_width, total_height),
        (left + slot_width + gap, top, slot_width, total_height),
    ]


def _get_main_plot_box(picture_boxes, slide_width, slide_height):
    """
    For main-plot layouts:
    - If template has a placeholder, expand horizontally
    - Otherwise use layout constants
    """
    if picture_boxes:
        _, top, _, height = picture_boxes[0]
        return (0, top, slide_width, height)

    box = MAIN_PLOT_BOX
    return (
        slide_width * box["left_ratio"],
        slide_height * box["top_ratio"],
        slide_width * box["width_ratio"],
        slide_height * box["height_ratio"],
    )


# ================================================================
# IMAGE INSERTION
# ================================================================

def _add_picture_fit(slide, image_path, left, top, width, height, fill_factor=1.0):
    """
    Insert image into slide, preserving aspect ratio and centering it.
    fill_factor > 1 expands slightly to avoid white bands.
    """
    image_path = str(image_path)
    shape = slide.Shapes.AddPicture(image_path, False, True, 0, 0, -1, -1)

    shape.LockAspectRatio = True

    scale = min(width / shape.Width, height / shape.Height)
    scale *= fill_factor

    shape.Width *= scale
    shape.Height *= scale

    shape.Left = left + (width - shape.Width) / 2
    shape.Top = top + (height - shape.Height) / 2

    # Cosmetic border to separate plots visually
    shape.Line.Visible = True
    shape.Line.ForeColor.RGB = 0
    shape.Line.Weight = 1

    return shape


# ================================================================
# TEMPLATE ASPECT RATIO EXTRACTION
# ================================================================

def get_template_plot_aspect_ratios(template_path, export_map):
    """
    Reads the PPTX template and extracts the native aspect ratios of picture
    placeholders so exported plots match layout proportions precisely.
    Returns empty dict if template cannot be read (graceful degradation).
    """
    template_path = Path(template_path).resolve()
    if not template_path.exists():
        print(f"[WARNING][powerpointexporter] PowerPoint template not found: {template_path}. Using default aspect ratios.")
        return {}

    aspect_ratios = {}

    try:
        with ZipFile(template_path) as pptx:
            pres_root = ET.fromstring(pptx.read("ppt/presentation.xml"))
            slide_size = pres_root.find("p:sldSz", PPTX_NS)
            slide_width = int(slide_size.attrib.get("cx", 0)) if slide_size is not None else None

            for slide_num, config in export_map.items():
                slide_xml = f"ppt/slides/slide{slide_num}.xml"
                if slide_xml not in pptx.namelist():
                    continue

                root = ET.fromstring(pptx.read(slide_xml))

                # Collect all <p:pic> shapes
                picture_boxes = []
                for pic in root.findall(".//p:pic", PPTX_NS):
                    xfrm = pic.find("p:spPr/a:xfrm", PPTX_NS)
                    if xfrm is None:
                        continue

                    ext = xfrm.find("a:ext", PPTX_NS)
                    off = xfrm.find("a:off", PPTX_NS)
                    if ext is None:
                        continue

                    width = int(ext.attrib.get("cx", 0))
                    height = int(ext.attrib.get("cy", 0))
                    left = int(off.attrib.get("x", 0)) if off is not None else 0
                    top = int(off.attrib.get("y", 0)) if off is not None else 0

                    if width > 0 and height > 0:
                        picture_boxes.append((left, top, width, height))

                picture_boxes.sort(key=lambda b: (b[0], b[1]))

                image_files = config.get("images", [])
                slide_aspects = []

                for i, img_file in enumerate(image_files):
                    if i >= len(picture_boxes):
                        break

                    _, _, w, h = picture_boxes[i]
                    # For main plot with only one picture, stretch horizontally
                    if (
                        config.get("layout") == "main_plot"
                        and slide_width is not None
                        and len(image_files) == 1
                    ):
                        w = slide_width  # stretch to full width

                    slide_aspects.append((img_file, w / h))

                # For two-up scatter plots → average aspect ratio
                if (
                    config.get("layout") == "double_plot"
                    and len(slide_aspects) == 2
                    and not all(
                        name.startswith(("scatter_", "psd_", "bar_"))
                        for name, _ in slide_aspects
                    )
                ):
                    avg = sum(a for _, a in slide_aspects) / len(slide_aspects)
                    for img, _ in slide_aspects:
                        aspect_ratios[img] = (avg,)
                else:
                    for img, ar in slide_aspects:
                        aspect_ratios[img] = ar
    except Exception as e:
        print(f"[WARNING][powerpointexporter] Error reading template aspect ratios: {e}. Using default aspect ratios.")
        return {}

    return aspect_ratios


# ================================================================
# MAIN EXPORT FUNCTION
# ================================================================

def export_report_to_powerpoint(template_path, output_path, plots_dir, export_map, visible=False):
    """
    Insert generated plots into a PowerPoint template according to export_map.
    """
    try:
        import win32com.client
    except ImportError as exc:
        raise ImportError(
            "pywin32 is required for PowerPoint export. Install with:\n"
            "    pip install pywin32"
        ) from exc

    template_path = Path(template_path).resolve()
    plots_dir = Path(plots_dir).resolve()
    output_path = Path(output_path).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template not found: {template_path}")

    # PowerPoint COM object
    ppt = win32com.client.Dispatch("PowerPoint.Application")
    ppt.Visible = True  # PowerPoint does not allow True/False control here

    pres = None
    try:
        pres = ppt.Presentations.Open(str(template_path), WithWindow=visible)

        slide_width = pres.PageSetup.SlideWidth
        slide_height = pres.PageSetup.SlideHeight

        for slide_num, cfg in export_map.items():
            slide = pres.Slides(slide_num)
            layout = cfg["layout"]
            image_list = cfg["images"]

            picture_boxes = _get_picture_boxes(slide)

            if layout == "main_plot" and len(image_list) == 1:
                target_boxes = [_get_main_plot_box(picture_boxes, slide_width, slide_height)]
            elif layout == "double_plot" and len(image_list) == 2:
                target_boxes = _get_double_plot_boxes(picture_boxes, slide_width, slide_height)
            else:
                target_boxes = picture_boxes or [
                    _resolve_box(layout, slide_width, slide_height, slot_index=i, slot_count=len(image_list))
                    for i in range(len(image_list))
                ]

            _replace_slide_pictures(slide)

            for i, img in enumerate(image_list):
                img_path = plots_dir / img
                if not img_path.exists():
                    print(f"[WARNING][powerpointexporter] Missing plot for slide {slide_num}: {img}")
                    continue

                if i < len(target_boxes):
                    left, top, width, height = target_boxes[i]
                else:
                    left, top, width, height = _resolve_box(
                        layout,
                        slide_width,
                        slide_height,
                        slot_index=i,
                        slot_count=len(image_list),
                    )

                # Aggressive padding for scatter/PSD in double-layout
                if (
                    layout == "double_plot"
                    and img.startswith(("scatter_", "psd_", "histogram_", "bar_"))
                ):
                    fill_factor = 1.2
                else:
                    fill_factor = 1.0

                _add_picture_fit(slide, img_path, left, top, width, height, fill_factor)

        # save result
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pres.SaveAs(str(output_path))
            final = output_path
        except Exception as exc:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = output_path.with_name(f"{output_path.stem}_{ts}{output_path.suffix}")
            print(
                f"[WARNING][powerpointexporter] Could not save to {output_path} ({exc}). Using fallback: {fallback}"
            )
            pres.SaveAs(str(fallback))
            final = fallback

        print(f"PowerPoint report saved to: {final}")

    except Exception as exc:
        print(f"[ERROR][powerpointexporter] PowerPoint export failed: {exc}")

    finally:
        try:
            ppt.Quit()
        except Exception as quit_err:
            print(f"[WARNING][powerpointexporter] Error quitting PowerPoint: {quit_err}")
        
        # Release COM objects
        pres = None
        ppt = None
