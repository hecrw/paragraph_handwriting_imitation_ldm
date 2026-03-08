from PIL import Image


class LazyImageDict:
    """
    Dict-like container that loads images from disk on access instead of
    holding every PIL Image in RAM.  Entries registered with set_path()
    are loaded lazily; entries assigned with []= are stored eagerly
    (backward-compat for code that constructs images dynamically).
    """

    def __init__(self):
        self._lazy = {}   # name -> (path, mode, crop_box | None)
        self._eager = {}  # name -> PIL.Image or np.ndarray

    def set_path(self, name, path, mode, crop_box=None):
        """Register an image to be loaded from *path* on first access."""
        self._lazy[name] = (path, mode, crop_box)

    # ---- dict-like interface ------------------------------------------------

    def __getitem__(self, name):
        if name in self._eager:
            return self._eager[name]
        path, mode, crop_box = self._lazy[name]
        img = Image.open(path).convert(mode)
        if crop_box is not None:
            img = img.crop(crop_box)
        return img

    def __setitem__(self, name, image):
        self._eager[name] = image

    def __contains__(self, name):
        return name in self._lazy or name in self._eager

    def __len__(self):
        return len(set(self._lazy.keys()) | set(self._eager.keys()))

    def keys(self):
        return set(self._lazy.keys()) | set(self._eager.keys())
