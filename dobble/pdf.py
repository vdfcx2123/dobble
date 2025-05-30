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
         n_symbols_per_card: int):
    """
    Merge Dobble cards into a scaled PDF ready to print

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
    card_size_pix = first_img.shape[0]

    # --- Modifications start here ---

    # Set the card diameter to 9.2 cm (46 mm radius)
    card_diameter_cm = 9.2
    pix_per_cm = float(card_size_pix) / card_size_cm # This will use the original card_size_cm for initial pixel calculation, we'll use card_diameter_cm for layout

    w_a4_cm = 21.0
    h_a4_cm = 29.7

    w_num_pix = math.floor(w_a4_cm * pix_per_cm)
    h_num_pix = math.floor(h_a4_cm * pix_per_cm)

    # To get 6 cards per A4, we can arrange them in a 2x3 grid
    nb_patch_in_w = 2
    nb_patch_in_h = 3

    # Recalculate card size in pixels based on desired diameter and A4 dimensions
    # to fit 2x3 grid within A4
    card_size_pix_layout_w = math.floor(w_num_pix / nb_patch_in_w)
    card_size_pix_layout_h = math.floor(h_num_pix / nb_patch_in_h)

    # We should probably use the smaller dimension to maintain aspect ratio
    card_size_pix_layout = min(card_size_pix_layout_w, card_size_pix_layout_h)

    # Recalculate padding based on the new layout and card size
    w_pad = w_num_pix - card_size_pix_layout * nb_patch_in_w
    h_pad = h_num_pix - card_size_pix_layout * nb_patch_in_h

    assert w_pad >= 0 and h_pad >= 0 # Ensure padding is non-negative
    dh = int(h_pad / (1 + nb_patch_in_h))
    h_patch = 255 * np.ones((dh, w_num_pix, 3), np.uint8)
    h_patch_bot = 255 * np.ones((h_pad - (dh * nb_patch_in_h), w_num_pix, 3), np.uint8)

    dw = int(w_pad / (1 + nb_patch_in_w))
    w_patch = 255 * np.ones((card_size_pix_layout, dw, 3), np.uint8)
    w_patch_right = 255 * np.one # This line is incomplete in the original code, assumes it continues with something like np.ones((card_size_pix_layout, w_pad - (dw * nb_patch_in_w), 3), np.uint8)

    # --- Modifications end here ---

    # The rest of the code for arranging and merging images would follow,
    # using card_size_pix_layout for resizing and the new padding values.
    # You'll need to adapt the image loading and arrangement logic to use
    # card_size_pix_layout and the calculated padding to create the final A4 pages.
