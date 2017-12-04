import os
from PIL import Image


class TestReader(object):
    def __init__(self, data_dir, transform=None):

        self.points_num = 14

        self.img_height = 512
        self.img_width = 512
        self.label_height = self.img_height >> 3
        self.label_width = self.img_width >> 3
        self.group_height = self.img_height >> 3
        self.group_width = self.img_width >> 3

        self.img_ids = list()
        self.img_paths = list()
        self.transform = transform

        for filename in os.listdir(data_dir):
            img_id = filename.split('.')[0]
            img_path = os.path.join(data_dir, filename)

            self.img_ids.append(img_id)
            self.img_paths.append(img_path)

        assert len(self.img_ids) == len(self.img_paths)

        print "Size of data for test: %d" % self.__len__()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        f_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size

        img_1 = self._resize(img, self.img_height, self.img_width)
        img_2 = self._resize(img, self.img_height + 128, self.img_width + 128)

        f_img_1 = self._resize(f_img, self.img_height, self.img_width)
        f_img_2 = self._resize(f_img, self.img_height + 128, self.img_width + 128)

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

            f_img_1 = self.transform(f_img_1)
            f_img_2 = self.transform(f_img_2)

        imgs = (
            img_1,
            img_2,
            f_img_1,
            f_img_2,
        )

        sample = (imgs, (self.label_height, self.label_width), (height, width),
                  self.img_ids[idx], self.img_paths[idx])

        return sample

    def _resize(self, img, img_height, img_width):
        out_img = img.resize((img_width, img_height), Image.BILINEAR)

        return out_img

