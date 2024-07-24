from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import bisect

@dataclass
class ImageDataEntry:
    timestamp: datetime
    image: Optional[Any] = None
    mask: Optional[Any] = None
    bbox: Optional[Any] = None

class ImageDataManager:
    """
    Images should be recieved sequentially.  So that a bbox/mask is not left outstanding without an image.
    """
    def __init__(self, skew: timedelta = timedelta(seconds=1)):
        self.data: List[ImageDataEntry] = []
        self.skew = skew
        self.latest_image_index = None
        self.latest_bbox_index = None
        self.latest_mask_index = None

    def add_image(self, image: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self.latest_image_index = self._add_entry(timestamp, image=image)

    def add_mask(self, mask: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self.latest_mask_index = self._add_entry(timestamp, mask=mask)

    def add_bbox(self, bbox: Any, timestamp: Optional[datetime] = None):
        timestamp = timestamp or datetime.now()
        self.latest_bbox_index = self._add_entry(timestamp, bbox=bbox)

    def _add_entry(self, timestamp, image=None, mask=None, bbox=None) -> int:
        """
        Return the index of the added entry
        """
        # First check if we have a new image to add and are not missing an image on the most recent entry
        if len(self) != 0 and image is not None and self.get_latest()["image"] is not None:
            # Set the next image
            self._insert_new_entry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
            return len(self)-1
        
        index = self._find_nearest_index(timestamp)
        if index is not None and abs(self.data[index].timestamp - timestamp) <= self.skew:
            existing_entry = self.data[index]
            if image is not None:
                existing_entry.image = image
            if mask is not None:
                existing_entry.mask = mask
            if bbox is not None:
                existing_entry.bbox = bbox
            return index
        else:
            self._insert_new_entry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
            if mask is not None or bbox is not None:
                print(f"Warning: Added mask or bbox with no nearby image within skew of {self.skew}")

            return len(self)-1

    def _insert_new_entry(self, image=None, mask=None, bbox=None, timestamp=None):
        new_entry = ImageDataEntry(timestamp=timestamp, image=image, mask=mask, bbox=bbox)
        bisect.insort(self.data, new_entry, key=lambda entry: entry.timestamp)

    def _find_nearest_index(self, timestamp):
        if not self.data:
            return None
        # timestamps = [entry.timestamp for entry in self.data]
        # index = bisect.bisect_left(timestamps, timestamp)
        index = bisect.bisect_left(self.data, timestamp, key=lambda entry: entry.timestamp)
        if index == len(self):
            index -= 1
        return index

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not self.data:
            raise IndexError("No data available.")
        entry = self.data[index]
        return {
            'timestamp': entry.timestamp,
            'image': entry.image,
            'mask': entry.mask,
            'bbox': entry.bbox
        }

    def __len__(self):
        return len(self.data)

    def get_latest(self) -> Dict[str, Any]:
        return self[-1]

    def get_latest_image(self) -> Dict[str, Any]:
        """
        :returns: None on get_latest_image.  May be the cause of a thrown error.
        """
        if self.latest_image_index is None:
            return None
        return self[self.latest_image_index]

    def get_latest_mask(self) -> Dict[str, Any]:
        """
        :returns: None on get_latest_image.  May be the cause of a thrown error.
        """
        if self.latest_mask_index is None:
            return None
        return self[self.latest_mask_index]

    def get_latest_bbox(self) -> Dict[str, Any]:
        """
        :returns: None on get_latest_image.  May be the cause of a thrown error.
        """
        if self.latest_bbox_index is None:
            return None
        return self[self.latest_bbox_index]

    def get_left(self, index) -> Dict[str, Any]:
        """
        Returns a dict with a image, bbox, and seg_map, looking left from idx until it finds them.
        """
        ret = self[index]
        while index > 0 and (ret["image"] is None or ret["mask"] is None or ret["bbox"] is None):
            index = index - 1
            entry = self[index]
            if ret["image"] is None and entry["image"] is not None:
                ret["image"] = entry["image"]
            if ret["mask"] is None and entry["mask"] is not None:
                ret["mask"] = entry["mask"]
            if ret["bbox"] is None and entry["bbox"] is not None:
                ret["bbox"] = entry["bbox"]

        return ret

