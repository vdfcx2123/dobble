
import math
import os

import cv2
import img2pdf
import numpy as np
from tqdm import tqdm

from dobble.profiling import profile
from dobble.utils import assert_len
from dobble.utils import list_image_files
from dobble.utils import new_folder


@profile

def main(cards_folder: str,
         out_print_folder: str,
         card_size_cm: float,
         n_symbols_per_card: int) -> None:
    """
    Merge Dobble cards into a scaled PDF ready to print, with 6 cards per A4 sheet.

    Args:
        cards_folder: Folder containing the high-res Dobble cards images
        out_print_folder: Output folder containing the batched cards and the PDF file
        card_size_cm: Diameter of the output Dobble cards to print
        n_symbols_per_card: Number of symbols per card
    """
    names = list_image_files(cards_folder)

    n_cards = n_symbols_per_card**2 - n_symbols_per_card + 1
    assert_len(names, n_cards)
    names += [None]  # Pad to have an even size

    pdf_path = os.path.join(out_print_folder, "cards.pdf")
    batches_folder = os.path.join(out_print_folder, "batches")
    new_folder(batches_folder)

    first_img = cv2.imread(os.path.join(cards_folder, names[0]))
    card_orig_pix = first_img.shape[0]

    pix_per_cm = float(card_orig_pix) / card_size_cm
    w_a4_cm = 21.0
    h_a4_cm = 29.7

    w_num_pix = math.floor(w_a4_cm * pix_per_cm)
    h_num_pix = math.floor(h_a4_cm * pix_per_cm)

    # Set fixed number of patches per row and column to 6 cards per page (2 x 3)
    nb_patch_in_w = 2  # cards across
    nb_patch_in_h = 3  # cards down

    # Calculate new card size to fit 2x3 grid on A4
    card_size_pix_w = w_num_pix / nb_patch_in_w
    card_size_pix_h = h_num_pix / nb_patch_in_h
    card_size_pix = int(min(card_size_pix_w, card_size_pix_h))

    # White padding patches (if needed — typically zero now)
    h_patch = 255 * np.ones((5, w_num_pix, 3), np.uint8)  # thin line separator
    w_patch = 255 * np.ones((card_size_pix, 5, 3), np.uint8)

    nb_patch_per_batch = nb_patch_in_w * nb_patch_in_h
    nb_of_batch = math.ceil(len(names) / nb_patch_per_batch)
    batches_paths = []

    for k in tqdm(range(nb_of_batch), "Batch cards"):
        batch_path = os.path.join(batches_folder, f"batch_cards_{k}.png")

        batch_images = [
            cv2.imread(os.path.join(cards_folder, name))
            if name is not None else 255 * np.ones_like(first_img)
            for name in names[nb_patch_per_batch * k: nb_patch_per_batch * (k + 1)]
        ]

        # Resize all images to new card size
        batch_images_resized = [
            cv2.resize(img, (card_size_pix, card_size_pix)) for img in batch_images
        ]

        columns = []
        idx = 0
        for col in range(nb_patch_in_h):
            line_imgs = []
            for row in range(nb_patch_in_w):
                if idx < len(batch_images_resized):
                    line_imgs.append(batch_images_resized[idx])
                else:
                    line_imgs.append(255 * np.ones((card_size_pix, card_size_pix, 3), np.uint8))
                if row < nb_patch_in_w - 1:
                    line_imgs.append(w_patch)
                idx += 1
            line_concat = cv2.hconcat(line_imgs)
            columns.append(line_concat)
            if col < nb_patch_in_h - 1:
                columns.append(h_patch)

        batch_img = cv2.vconcat(columns)
        cv2.imwrite(batch_path, batch_img)
        batches_paths.append(batch_path)

    a4inpt = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
    layout_fun = img2pdf.get_layout_fun(a4inpt)
    with open(pdf_path, "wb") as f:
        f.write(img2pdf.convert(batches_paths, layout_fun=layout_fun))

    print(f"✅ Dobble cards PDF with 6 cards per A4 saved at:\n{os.path.abspath(pdf_path)}")
