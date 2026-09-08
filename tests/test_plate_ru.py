# -*- coding: utf-8 -*-
"""
Правила формата номера РФ.

Запуск (без доп. пакетов, pytest не нужен):
    python -m unittest discover -s tests -v

Эти проверки — граница между «номер» и «фантом»: без валидации региона в БД
уезжали чтения вида Х034ХА356, где такого региона не существует.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plate_ru  # noqa: E402


class NormalizeTests(unittest.TestCase):
    def test_normalize_plate(self):
        cases = [
            ("a636aa06", "А636АА06"),
            ("В 713 ВВ 125", "В713ВВ125"),
            ("t314xc125", "Т314ХС125"),
            ("", ""),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(plate_ru.normalize_plate(raw), expected)


class FormatTests(unittest.TestCase):
    def test_classification(self):
        cases = [
            ("А636АА06", True, False),
            ("В713ВВ125", True, False),
            ("9036СС45", False, True),
            ("АБВ123", False, False),
        ]
        for plate, civ, mil in cases:
            with self.subTest(plate=plate):
                self.assertIs(plate_ru.is_civilian(plate), civ)
                self.assertIs(plate_ru.is_military(plate), mil)

    def test_region_validation(self):
        cases = [
            ("В713ВВ125", True),
            ("А636АА06", True),
            ("9036СС45", True),
            ("Х034ХА356", False),   # региона 356 не существует
            ("В713ВВ107", False),
        ]
        for plate, valid in cases:
            with self.subTest(plate=plate):
                self.assertIs(plate_ru.region_is_valid(plate), valid)


class DecodeConstrainedTests(unittest.TestCase):
    def test_exact_match_passes_through(self):
        self.assertEqual(
            plate_ru.decode_constrained("В713ВВ125"), ("В713ВВ125", "civilian"))

    def test_fixes_glyph_class(self):
        """Буква на месте цифры — типичная путаница CTC-головы: О->0, Т->7."""
        self.assertEqual(
            plate_ru.decode_constrained("ВО13ВВ125"), ("В013ВВ125", "civilian"))
        # Гражданская маска требует одной подмены (Т->7), военная — двух (В->8, Т->7)
        self.assertEqual(
            plate_ru.decode_constrained("ВТ13ВВ125"), ("В713ВВ125", "civilian"))

    def test_prefers_exact_military_over_fitted_civilian(self):
        """8713ВВ125 сам по себе валидный военный номер — подменять В не нужно."""
        self.assertEqual(
            plate_ru.decode_constrained("8713ВВ125"), ("8713ВВ125", "military"))

    def test_strips_glued_letter_before_military(self):
        self.assertEqual(
            plate_ru.decode_constrained("Е9036СС45"), ("9036СС45", "military"))

    def test_rejects_garbage(self):
        for text in ("", "АБВ", "1", "ВВВВВВВВВ"):
            with self.subTest(text=text):
                self.assertIsNone(plate_ru.decode_constrained(text))

    def test_rejects_invalid_region(self):
        self.assertIsNone(plate_ru.decode_constrained("В713ВВ356"))
        self.assertIsNotNone(
            plate_ru.decode_constrained("В713ВВ356", require_region=False))


class VoteTests(unittest.TestCase):
    def test_picks_majority(self):
        voted, conf, votes = plate_ru.vote_plate([
            ("В713ВВ125", [0.9] * 9),
            ("В713ВВ125", [0.9] * 9),
            ("В713НВ125", [0.9] * 9),
        ])
        self.assertEqual(voted, "В713ВВ125")
        self.assertEqual(votes, 3)
        self.assertGreater(conf, 0.5)

    def test_weighs_by_confidence(self):
        voted, _, _ = plate_ru.vote_plate([
            ("Х034ХА125", [0.98] * 9),
            ("Х084ХА125", [0.30] * 9),
        ])
        self.assertEqual(voted, "Х034ХА125")

    def test_does_not_mix_lengths(self):
        voted, _, votes = plate_ru.vote_plate([
            ("А123ВС125", [0.9] * 9),
            ("А123ВС125", [0.9] * 9),
            ("А123ВС12", [0.9] * 8),
        ])
        self.assertEqual(voted, "А123ВС125")
        self.assertEqual(votes, 2)

    def test_empty(self):
        self.assertIsNone(plate_ru.vote_plate([]))


class LetterboxTests(unittest.TestCase):
    def test_letterbox_keeps_canvas_size(self):
        try:
            import numpy as np
            from nomeroff_net.tools.image_processing import letterbox_resize
        except Exception:
            self.skipTest("cv2/nomeroff_net недоступны")
        img = np.zeros((20, 80, 3), dtype=np.uint8)
        out = letterbox_resize(img, 200, 50)
        self.assertEqual(out.shape[:2], (50, 200))


if __name__ == "__main__":
    unittest.main()
