import os
from .decimer_segmentation import segment_chemical_structures_from_file
from .decimer_segmentation import save_images, save_images_pil, get_square_image

def dected_img(
    pdf_path,
    save_path
):
    segments = segment_chemical_structures_from_file(pdf_path)
    raw_segments = [segs[0] for segs in segments]
    org_segments = [segs[1] for segs in segments]
    pdf_name = os.path.basename(pdf_path)[:-4]
    norm_dir = os.path.join(save_path, pdf_name, "norm")
    org_dir = os.path.join(save_path, pdf_name, "org")
    if not os.path.exists(norm_dir):
        os.makedirs(norm_dir)
    if not os.path.exists(org_dir):
        os.makedirs(org_dir)
    # save_images(
    #     raw_segments, segment_dir, f"{pdf_name}_orig"
    # )
    # binarized_segments = [get_bnw_image(segment) for segment in raw_segments]
    # save_images(
    #     binarized_segments, segment_dir, f"{pdf_name}_bnw"
    # )
    normalized_segments = [
        get_square_image(segment, 299) for segment in raw_segments
    ]
    save_images(
        normalized_segments,
        norm_dir,
        f"{pdf_name}_norm",
    )
    save_images_pil(
        org_segments,
        org_dir,
        f"{pdf_name}_org",
    )


# if __name__ == "__main__":
#     pdf_path = r"/root/MW/dected_pdf/test_page.pdf"
#     save_path = r"/root/MW/dected_pdf/test_page"
#     raw_segments = segment_chemical_structures_from_file(pdf_path)
#     pdf_name = os.path.basename(pdf_path)[:-4]
#     segment_dir = os.path.join(save_path, pdf_name, "segments")
#     if not os.path.exists(segment_dir):
#         os.makedirs(segment_dir)
#     save_images(
#         raw_segments, segment_dir, f"{pdf_name}_orig"
#     )
#     binarized_segments = [get_bnw_image(segment) for segment in raw_segments]
#     save_images(
#         binarized_segments, segment_dir, f"{pdf_name}_bnw"
#     )
#     normalized_segments = [
#         get_square_image(segment, 299) for segment in raw_segments
#     ]
#     save_images(
#         normalized_segments,
#         segment_dir,
#         f"{pdf_name}_norm",
#     )
    
    