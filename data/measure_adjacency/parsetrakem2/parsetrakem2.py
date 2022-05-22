"""
parsetrakem2.py
A module containing classes used to parse TrakEM2 files
Required 3rd party packages:
  lxml
  numpy
  scipy
Author: Christopher Brittin
"""
import lxml.etree as etree
import itertools
import numpy as np
from scipy.spatial.distance import cdist


class ParseTrakEM2(object):
    """
    Class used to represent a TrakEM2 file.
    ...
    Attributes
    ---------
    trakem2 : str
       path to trakem2 file
    xml : etree.parse
       Parsed object for trakem2 file
    layers : dictionary
       Dictionalary of layer objects,
       (key=layer name,value=Layer(object))
    area_lists : dictionary
       Dictionary of area lists, (key=cell name, value = AreaList(object))
    dx : int
       transform applied to x
    dy : int
       transform applied to y
    Methods
    -------
    get_layers()
      Assigns dictionary of Layer(objects) to self.layers
    get_area_lists()
      Assigns dictionary of AreaList(objects) to self.area_lists
    set_fill(color,opacity=0.5)
      Changes the fill color for of area list.
    get_calibration(self)
      Sets the transforms for self.dx and self.dy
    get_boundaries_in_layer(layer,scale_bounding_box=1,area_thresh=200,**kwargs)
      Returns a dictionary of boundaries for the area lists in layer.
    get_overlapping_boundiaries(boundaries)
      Returns list of boundaries with overlapping bounding boxes
    is_boundary_overlap(A,B)
      returns True if boundaries A and B have overlapping bounding boxes.
    compute_adjacency(A,B,pixel_radius=10)
      return the length of adjacency (int) between boundaries A and B
    batch_compute_adjacency(boundaries,pixel_radius=10)
      returns lenth of adjacencies for a list of bondary pairs
    """

    def __init__(self, trakem2):
        """
        Parameters:
        ----------
        trakem2 : str
           path to trakem2 file
        """

        self.trakem2 = trakem2
        parser = etree.XMLParser(remove_blank_text=True)
        self.xml = etree.parse(trakem2, parser)
        self.layers = None
        self.area_list = None
        self.dx = 0
        self.dy = 0

    def get_layers(self):
        """
        Assigns dictionary of Layer(objects) to self.layers

        self.layers is a dictionary
        (key=layer name, val=Layer(object))
        """

        layer_width = self.xml.xpath("//t2_layer_set/@layer_width")[0]
        layer_height = self.xml.xpath("//t2_layer_set/@layer_height")[0]
        self.layer_dim = [float(layer_width), float(layer_height)]
        oid = self.xml.xpath("//t2_layer/@oid")
        thickness = self.xml.xpath("//t2_layer/@thickness")
        z = self.xml.xpath("//t2_layer/@z")
        title = self.xml.xpath("//t2_patch/@title")
        trans = self.xml.xpath("//t2_patch/@transform")
        width = self.xml.xpath("//t2_patch/@width")
        height = self.xml.xpath("//t2_patch/@height")
        self.layers = {}
        for i in range(len(oid)):
            temp = trans[i].split(',')
            temp = (float(temp[-2]), float(temp[-1].replace(')', '')))
            L = Layer(title[i].replace('.tif', ''))
            L.oid = oid[i]
            L.transform = temp
            L.height = height[i]
            L.width = width[i]
            L.thickness = thickness[i]
            L.z = float(z[i])
            self.layers[L.name] = L

    def get_area_lists(self):
        """
        Assigns dictionary of AreaList(objects) to self.area_lists
        self.area_lists is a dictionary
        (key=cell name, val=AreaList(object))
        """
        area_lists = self.xml.xpath("//t2_area_list/@title")
        trans = self.xml.xpath("//t2_area_list/@transform")
        self.area_lists = {}
        for i in range(len(area_lists)):
            temp = trans[i].split(',')
            temp = [int(float(temp[-2])),
                    int(float(temp[-1].replace(')', '')))]
            temp[0] -= self.dx
            temp[1] -= self.dy
            temp = tuple(temp)
            A = AreaList(area_lists[i])
            A.transform = temp
            self.area_lists[A.name] = A
        if 'area_list' in self.area_lists.keys():
            del self.area_lists['area_list']

    def set_fill(self, colors, opacity=0.5):
        """
        Changes the fill color for of area list.

        Parameters
        ----------
        colors : list
          List of tuples [(cell_name1,color1), (cell_name2, color2)...]
          where for each tuple, the first entry is the area list name
          and the second entry is an HTML color code e.g. #000000
        opacity : float
          Opacity of the fill. (default is 0.5)

        """
        root = self.xml.getroot()
        for (neuron, color) in colors:
            area_list = root.find(".//t2_area_list[@title='%s']" % neuron)
            style = ("stroke:none;fill-opacity:%1.2f;fill:%s;"
                     % (opacity, color.lower()))
            area_list.attrib['style'] = style
        etree.tostring(self.xml)

    def get_calibration(self):
        self.px_width = float(self.xml.xpath("//t2_calibration/@pixelWidth")[0])
        self.px_height = float(self.xml.xpath("//t2_calibration/@pixelHeight")[0])
        self.px_depth = float(self.xml.xpath("//t2_calibration/@pixelDepth")[0])
        trans = self.xml.xpath("//t2_patch/@transform")[0]
        trans = trans.replace(')', '')
        trans = trans.split(',')
        self.dx = float(trans[-2])
        self.dy = float(trans[-1])

    def get_boundaries_in_layer(self, layer, scale_bounding_box=1,
                                area_thresh=200,
                                area_lists=None):
        """
        Returns a dictionary of boundaries for the area lists in layer.
        Parameters
        ----------

        layers : str
          Layer name
        scale_bounding_box : float
          Scales the bounding boxes of the area list boundaries if not
          equal to 1. (default is 1)
        area_thresh : int
          Area lists with areas less than area_thresh are not returned
          (default is 200 px^2)
        area_lists : list
         Specifies which area lists to include. Should be a list of
         area list names. If not specified will use all area lists
         in the layer. (default is None)


        Returns
        ----------
        boundary : 2D dictionary
         A 2D dictionary of Boundary(object) i.e.
         boundary[name] = {0:Boundary(object), 1:Boundary(object)...}
        """
        area_lists = []
        layer = self.layers[layer]
        if not area_lists: area_lists = self.area_lists.keys()
        boundary = {}
        for n in area_lists:
            path = ("//t2_area_list[@title='%s']"
                    "/t2_area[@layer_id='%s']"
                    "/t2_path/@d" % (n, layer.oid))
            path = self.xml.xpath(path)
            temp, idx = {}, 0
            for p in path:
                p = self.area_lists[n].path_transform(p)
                b = Boundary(n, idx, p, transform=layer.transform)
                b.set_area()
                if b.area > area_thresh:
                    b.fill_boundary_gaps()
                    b.set_bounding_box()
                    if scale_bounding_box != 1:
                        b.scale_bounding_box(scale_bounding_box)
                    temp[idx] = b
                    idx += 1
            if temp:
                boundary[n] = temp

        return boundary

    def get_overlapping_boundaries(self, boundaries):
        """
        Returns list of boundaries with overlapping bounding boxes
        Parameters
        ----------
        boundaries : list
           List of tuples: [(B1,B2),(B1,B3),(B2,B3)...]
           Where B1,B2,B3,etc are Boundary(objects)

        Returns
        ----------
        overlaps : list
           List of tuples [(B1,B2),(B1,B3),(B2,B3)...]
           Where B1,B2,B3,etc are Boundary(objects) with bounding boxes
           that overlap.
        """

        nlst = [n for n in boundaries if n not in ['Pharynx', 'Phi_Marker']]
        comb = itertools.combinations(nlst, 2)
        overlaps = []
        for (a, b) in comb:
            for i in boundaries[a]:
                for j in boundaries[b]:
                    if self.is_boundary_overlap(boundaries[a][i], boundaries[b][j]):
                        overlaps.append((boundaries[a][i], boundaries[b][j]))
        return overlaps

    def is_boundary_overlap(self, A, B):
        """
        Return the length of adjacency (int) between boundaries A and B

        Parameters
        ----------
        A : Boundary(object)
        B : Boundary(object)

        Returns
        -------
        bool : bool
          True is boundaries overlap, False otherwise

        """
        [amin, amax] = A.bounding_box
        [bmin, bmax] = B.bounding_box
        # if one bounding box is on the left side of the other
        if (bmin[0] > amax[0] or amin[0] > bmax[0]):
            return False
        # If one bounding box is above another
        if (bmin[1] > amax[1] or amin[1] > bmax[1]):
            return False

        return True

    def compute_adjacency(self, A, B, pixel_radius=10):
        """
        Returns the length of adjacency (int) between boundaries A and B

        Parameters
        ----------
        A : Boundary(object)
        B : Boundary(object)
        pixel_radius : int
          Boundary points closer than the pixel radius are classified
          as adjacent. (default is 10)

        Returns
        ----------
        adj : int
           Length of adjacency = min(lA,lB) where lA is the number
           of pixels in boundary A adjacent to B and lB is the number
           of pixels in boundary B adjacent to A.
        """

        XA = np.array(A.path)
        XB = np.array(B.path)
        Y = cdist(XA, XB, 'euclidean')
        I = np.where(Y <= pixel_radius)
        adj = min(len(set(I[0])), len(set(I[1])))
        return adj

    def batch_compute_adjacency(self, boundaries, pixel_radius=10):
        """
        Returns lenth of adjacencies for a list of bondary pairs

        Parameters
        ----------
        boundaries : list
          List of boundary objects [(B1,B2),(B1,B3),(B2,B3)...]
          Where B1,B2,B3,etc. are boundary objects
        pixel_radius : int
          Boundary points closer than the pixel radius are classified
          as adjacent. (default is 10)
        Returns
        ---------
        adj : list
          list of adjacencies for boundary pairs
          [(B1,B2,adj_12),(B1,B3,adj_13),....]


        """
        adj = []
        for (b1, b2) in boundaries:
            a = self.compute_adjacency(b1, b2, pixel_radius=pixel_radius)
            if a > 0:
                adj.append((b1, b2, a))
        return adj


class Layer(object):
    """
    Class used to hold layer information

    ....
    Attributes
    ----------
    name : str
      name of layer
    oid  : int
      TrakEM2 //t2_layer/@oid
    transform : list
      TrakEM2 //t2_patch/@transform
    height : int
      TrakEM2 //t2_patch/@height
    width : int
      TrakEM2 //t2_patch/@width
    z : float
      TrakEM2 //t2_layer/@z
    """

    def __init__(self, name):
        self.name = name
        self.oid = None
        self.transform = None
        self.height = None
        self.width = None
        self.thickness = None
        self.z = None


class AreaList(object):
    """
    Class to used to hold area list info

    Attributes
    ----------
    name : str
      area list name
    transform : list
      TrakEM2 //t2_area_list/@transform
    """

    def __init__(self, name):
        self.name = name
        self.transform = None

    def path_transform(self, path):
        """
        Applies a x,y translation to the path determined by
        self.transform

        Parameters
        ----------
        path : Boundary(object).path
          Boundary path
        Returns
        ---------
        path : list
          Return a tranformed boundary object path
        """

        path = path.replace('M ', '')
        path = path.replace(' z', '')
        path = path.split(' L ')
        path = [list(map(float, p.split(' '))) for p in path]
        path = [(p[0] + self.transform[0], p[1] + self.transform[1])
                for p in path]
        return path


class Boundary(object):
    """
    Class used to represent boundary objects
    Attributes
    ----------
    name : str
     name of cell
    index : int
     index of cell boundary in the given layer
    path : list of tuples
     Path extracted from TrakEM2 file
    transform : tuple
     Transform to be applied to path
    area : int
     area enclosed by the boundary
    cent : int
     centroid of boundary
    boundary_length : int
     Number of pixels in boundary
    bounding_box : list
     List of min and max points of bounding box [(xmin,ymin),(xmax,ymax)]
    width : int
     Width of bounding box
    height : int
     Height of bounding box
    Methods
    ---------
    set_centroid()
      Computes the centroid of the boundary
    set_boundary_length()
      Computes number of pixels in the boundary
    set_area()
      Computes the area enclosed by the boundary
    set_bounding_box()
      Sets the bounding box parameters
    scale_bounding_box(scale)
      Scales the bounding box about the center
    fill_boundary_gaps()
      Fills any gaps in the boundary path to make it continous
    get_display_matrix()
      Returns a matrix A with the dimensions of the bounding box. A[i,j] = 1 if
      it is a boundary point and A[i,j] = 0 otherwise.

    """

    def __init__(self, name, index, path, **kwargs):
        self.name = name
        self.index = index
        self.path = path
        self.transform = (0, 0)
        self.area = None
        self.cent = None
        if 'transform' in kwargs:
            self.transform = kwargs['transform']

    def set_centroid(self):
        """
        Computes the centroid of the boundary

        Subtracting the transform puts centroid into
        Elegance DB coordinates
        """

        cent = [list(t) for t in zip(*self.path)]
        self.cent = [np.mean(cent[0]) - self.transform[0],
                     np.mean(cent[1]) - self.transform[1]]

    def set_boundary_length(self):
        """
        Computes number of pixels in the boundary
        """

        self.boundary_length = len(self.path)

    def set_area(self):
        """
        Computes the area enclosed by the boundary
        """
        B = self.path
        # Computes area of polygon defined by given boundary 'B'
        area = 0
        idx = [(i, i - 1) for i in range(len(B))]
        for (i, j) in idx:
            # Algorithm taken from http://alienryderflex.com/polygon_area
            # temp = (B[j][0] + B[i][0])*(B[j][1] - B[i][1])
            # Definition of area of polygon
            area += (B[i][0] * B[j][1] - B[j][0] * B[i][1])
        self.area = 0.5 * abs(area)

    def set_bounding_box(self):
        """
        Sets the bounding box parameters
        """
        x_coordinates, y_coordinates = zip(*self.path)
        self.bounding_box = [(min(x_coordinates), min(y_coordinates)),
                             (max(x_coordinates), max(y_coordinates))]
        self.width = self.bounding_box[1][0] - self.bounding_box[0][0] + 1
        self.height = self.bounding_box[1][1] - self.bounding_box[0][1] + 1

    def scale_bounding_box(self, scale):
        """
        Scales the bounding box about the center
        Parameters
        ----------
        scale : float
          Greater than 1 increases bounding box, less than 1 decreases.
        """
        [(minx, miny), (maxx, maxy)] = self.bounding_box
        xradius = self.width / 2
        yradius = self.height / 2
        xradius = np.ceil(scale * xradius)
        yradius = np.ceil(scale * yradius)
        minx = int(max(0, minx - xradius))
        miny = int(max(0, miny - yradius))
        maxx = int(maxx + xradius)
        maxy = int(maxy + yradius)
        self.bounding_box = [(minx, miny), (maxx, maxy)]
        self.width = self.bounding_box[1][0] - self.bounding_box[0][0] + 1
        self.height = self.bounding_box[1][1] - self.bounding_box[0][1] + 1

    def fill_boundary_gaps(self):
        """
        Fills any gaps in the boundary path to make it continous
        """
        zB = list(zip(self.path[:-1], self.path[1:]))
        zB.append((self.path[-1], self.path[0]))
        cnts = []
        for (c1, c2) in zB:
            c1, c2 = list(map(int, c1)), list(map(int, c2))
            lr, ud = [], []
            cnts.append(tuple(c1))
            # move left or right
            if c1[1] > c2[1]:
                lr = range(c2[1], c1[1] + 1)
            elif c2[1] > c1[1]:
                lr = range(c1[1], c2[1] + 1)
            for y in lr:
                cnts.append((c1[0], y))
            # move up or down
            if c1[0] > c2[0]:
                ud = range(c2[0], c1[0] + 1)
            elif c2[0] < c1[0]:
                ud = range(c1[0], c2[0] + 1)
            for x in ud:
                cnts.append((x, c2[1]))
        self.path = cnts

    def get_display_matrix(self):
        """
        Returns a matrix A with the dimensions of the bounding box.
        A[i,j] = 1 if it is a boundary point and A[i,j] = 0 otherwise.
        """
        [(minx, miny), (maxx, maxy)] = self.bounding_box
        A = np.zeros([self.height, self.width])
        for (x, y) in self.path:
            j = x - minx
            i = y - miny
            A[i, j] = 1
        return A


