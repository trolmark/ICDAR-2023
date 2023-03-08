import tensorflow as tf
import math
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
from shapely.ops import unary_union
from shapely.ops import nearest_points
from shapely.validation import make_valid
import constants as cnt



PEAK_EPSILON = 1e-6

# Taken from https://github.com/talmolab/sleap/tree/develop
def find_local_peaks_rough(cms: tf.Tensor, threshold: float = 0.1):
  """Find the global maximum for each sample and channel.
  Args:
      cms: Tensor of shape (samples, height, width, channels).
      threshold: Scalar float specifying the minimum confidence value for peaks. Peaks
          with values below this threshold will be replaced with NaNs.
  Returns:
      A tuple of (peak_points, peak_vals, peak_sample_inds, peak_channel_inds).
      peak_points: float32 tensor of shape (n_peaks, 2), where the last axis
      indicates peak locations in xy order.
      peak_vals: float32 tensor of shape (n_peaks,) containing the values at the peak
      points.
      peak_sample_inds: int32 tensor of shape (n_peaks,) containing the indices of the
      sample each peak belongs to.
      peak_channel_inds: int32 tensor of shape (n_peaks,) containing the indices of
      the channel each peak belongs to.
  """
  # Build custom local NMS kernel.
  kernel = tf.reshape(
      tf.constant([[0, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=tf.float32), (3, 3, 1)
  )

  # Reshape to have singleton channels.
  height = tf.shape(cms)[1]
  width = tf.shape(cms)[2]
  channels = tf.shape(cms)[3]
  flat_img = tf.reshape(tf.transpose(cms, [0, 3, 1, 2]), [-1, height, width, 1])

  # Perform dilation filtering to find local maxima per channel and reshape back.
  max_img = tf.nn.dilation2d(
      flat_img, kernel, [1, 1, 1, 1], "SAME", "NHWC", [1, 1, 1, 1]
  )
  max_img = tf.transpose(
      tf.reshape(max_img, [-1, channels, height, width]), [0, 2, 3, 1]
  )

  # Filter for maxima and threshold.
  argmax_and_thresh_img = (cms > max_img) & (cms > threshold)

  # Convert to subscripts.
  peak_subs = tf.where(argmax_and_thresh_img)

  # Get peak values.
  peak_vals = tf.gather_nd(cms, peak_subs)

  # Convert to points format.
  peak_points = tf.cast(tf.gather(peak_subs, [2, 1], axis=1), tf.float32)

  # Pull out indexing vectors.
  peak_sample_inds = tf.cast(tf.gather(peak_subs, 0, axis=1), tf.int32)
  peak_channel_inds = tf.cast(tf.gather(peak_subs, 3, axis=1), tf.int32)

  return peak_points, peak_vals, peak_sample_inds, peak_channel_inds


def find_peaks(cms, normalize = False):
  """Run peak finding on predicted confidence maps."""
  # Find local peaks.
  (
    peaks,
    peak_vals,
    peak_sample_inds,
    peak_channel_inds,
  ) = find_local_peaks_rough(cms)

  if normalize:
    h, w = tf.cast(cnt.INPUT_SIZE, dtype = tf.float32), tf.cast(cnt.INPUT_SIZE, dtype = tf.float32)
    peaks /= tf.stack([w, h])
  
  # Group peaks by sample.
  n_samples = tf.shape(cms)[0]
  peaks = tf.RaggedTensor.from_value_rowids(
      peaks, peak_sample_inds, nrows=n_samples
  )
  peak_vals = tf.RaggedTensor.from_value_rowids(
      peak_vals, peak_sample_inds, nrows=n_samples
  )
  peak_channel_inds = tf.RaggedTensor.from_value_rowids(
      peak_channel_inds, peak_sample_inds, nrows=n_samples
  )

  peaks_tensor = peaks.to_tensor()
  peak_vals_tensor = peak_vals.to_tensor()

  # Pad with zeros
  if cnt.MAX_KEYPOINTS_ON_IMAGE > tf.shape(peaks_tensor)[1]:
    padding_add = cnt.MAX_KEYPOINTS_ON_IMAGE - tf.shape(peaks_tensor)[1]
    peaks_tensor = tf.pad(peaks_tensor, tf.stack([[0, 0], [0, padding_add], [0, 0]]))

  return peaks_tensor, peak_vals_tensor


def group_peaks_in_polygon(peaks, pil_image):

  """
  Group raw peaks for image into polygon 
  """

  def fill_image_polygon():
    """Return polygon which contains whole image."""
    top_left = (1, 1)
    top_right = (pil_image.width - 1, 1)
    bottom_right = (pil_image.width - 1, pil_image.height - 1)
    bottom_left = (1, pil_image.height - 1)
    return [top_left, top_right, bottom_right, bottom_left]

  def project_point_on_line(point, line):
    """Return polygon which contains whole image."""
    x = np.array(point.coords[0])
    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)
    P = u + n*np.dot(x - u, n)
    return P

  def polar_point(center, angle, radius):
    radians = math.radians(angle)
    x = radius * math.cos(radians)
    y = radius * math.sin(radians)
    return (int(center[0] + x), int(center[1] + y))

  def sort_points_by_angle(center, points):
    cx, cy = center[0], center[1]
    x, y = np.array(points).T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(angles)
    return np.array(points)[indices]

  def merge_points_in_arc(points):
    distance_between_points = []
    for i in range(len(points) - 1):
      distance = Point(points[i]).distance(Point(points[i + 1]))
      distance_between_points.append(distance)

    # TODO: adding 1 important for correct display
    max_distance_idx = np.argsort(distance_between_points)[-1] + 1
    points = np.concatenate((points[max_distance_idx:], points[:max_distance_idx]))
    return points


  side = max(pil_image.width, pil_image.height)
  center = (pil_image.width // 2, pil_image.height // 2)

  internal_points = []
  external_points = []

  min_distance_between_ray_and_point = 10
  mean_interpoint_distance = 0
  detected_keypoints_threshold = 17
 
  # General idea:
  # 1. Pass rays from center
  # 2. Find points that are close to the ray
  # 3. Sort them in internal and external contour
  # 
  for angle in range(0, 360, 10):
    border_point = polar_point(center, angle, side)
    ray = LineString([center, border_point])

    close_peaks = [peak for peak in peaks if ray.distance(Point(peak)) < min_distance_between_ray_and_point]
    close_peaks = [project_point_on_line(Point(point), ray) for point in close_peaks]
    if len(close_peaks) < 2:
      continue

    close_peaks = close_peaks[:2]

    ## Find close points to center
    distance_to_center = [Point(peak).distance(Point(center)) for peak in close_peaks]
    distance_to_center_idx = np.argsort(distance_to_center)
    internal_idx, external_idx = distance_to_center_idx[0], distance_to_center_idx[1]

    internal_point = close_peaks[internal_idx]
    external_point = close_peaks[external_idx]

    interpoint_distance = Point(external_point).distance(Point(internal_point))
    if mean_interpoint_distance == 0 and interpoint_distance > 0.01 * side:
      mean_interpoint_distance = interpoint_distance

    # Filter too far points and too close points 
    # Heuristics that worked well for most cases: needed 
    if interpoint_distance < 0.5 * mean_interpoint_distance or\
      interpoint_distance > 4 * mean_interpoint_distance:
      continue

    mean_interpoint_distance += interpoint_distance
    mean_interpoint_distance /= 2
    
    internal_points.append(internal_point)
    external_points.append(external_point)

  if len(internal_points) < 2:
    return pil_image, [], [], fill_image_polygon()

  # Sort by angle
  sorted_internal = sort_points_by_angle(center, internal_points)
  sorted_external = sort_points_by_angle(center, external_points)

  # Merge points in arc and find endpoints
  sorted_external = merge_points_in_arc(sorted_external)
  sorted_internal = merge_points_in_arc(sorted_internal)

  # Combine arcs in polygon
  sorted_internal = [(x[0], x[1]) for x in sorted_internal]
  sorted_external = [(x[0], x[1]) for x in sorted_external]

  polygon_points  = list(reversed(sorted_internal)) + list(sorted_external)

  # Validation checks
  polygon = Polygon(polygon_points)
  if not polygon.is_valid:
    polygon_points = fill_image_polygon()

  if len(polygon_points) < detected_keypoints_threshold:
    polygon_points = fill_image_polygon()

  return pil_image, sorted_internal, sorted_external, polygon_points
  
