from contextlib import contextmanager


class ImageSizeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size_x = None
        self.image_size_y = None

    def set_working_image_size(self, size):
        size_x, size_y = size
        self.image_size_x = size_x
        self.image_size_y = size_y


class ImageSizeManagerMixin(ImageSizeMixin):
    def set_image_size(self, sizes):
        def work(module):
            if isinstance(module, ImageSizeMixin):
                module.set_working_image_size(sizes)

        self.apply(work)

    @contextmanager
    def with_image_size(self, sizes):
        old_sizes = (self.image_size_x, self.image_size_y)
        self.set_image_size(sizes)
        try:
            yield
        finally:
            self.set_image_size(old_sizes)
