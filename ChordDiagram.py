# -*- coding: utf-8 -*-
"""
Chord Diagrams for Bokeh

Author: Niklas Berliner
"""

import numpy  as np

from operator  import itemgetter

from bokeh          import palettes
from bokeh.plotting import figure
from bokeh.models   import Range1d


class Color():
    """ Make it easy to iterate through different colors. """
    def __init__(self):
        
        # Define the default color palette. The current one contains only 7
        # different colors after which the first one is reused. Check
        # for more color palettes;
        # http://bokeh.pydata.org/en/latest/docs/reference/palettes.html
        self.palette = palettes.Set2[7]
        self.i = -1
    
    def __call__(self):
        self.i += 1
        if self.i >= len(self.palette):
            self.i = 0
        return(self.palette[self.i])
    
    def hex_to_rgb(self, value): # http://stackoverflow.com/a/214657/1922650
        """Return (red, green, blue) for the color given as #rrggbb."""
        value = value.lstrip('#')
        lv = len(value)
        return(tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)))
    
    def rgb_to_hex(self, red, green, blue):
        """Return color as #rrggbb for the given color values."""
        return('#%02x%02x%02x' % (red, green, blue))
        
    def rgba_to_rgb(self, red, green, blue, alpha):
        # http://stackoverflow.com/a/2049362/1922650
        source = np.rec.array([(red/255.,green/255.,blue/255.,alpha), ], 
                              dtype=[('red', 'float'),('green', 'float'), ('blue', 'float'), ('alpha', 'float')])
        
        bground = np.rec.array([(1.,1.,1.,1.), ], 
                              dtype=[('red', 'float'),('green', 'float'), ('blue', 'float'), ('alpha', 'float')])
    
        red   = ((1. - source[0].alpha) * bground[0].red)   + (source[0].alpha * source[0].red)
        green = ((1. - source[0].alpha) * bground[0].green) + (source[0].alpha * source[0].green)
        blue  = ((1. - source[0].alpha) * bground[0].blue)  + (source[0].alpha * source[0].blue)
        
        red   *= 255.
        green *= 255.
        blue  *= 255.
        
        if red > 255:
            red = 255
        if green > 255:
            green = 255
        if blue > 255:
            blue = 255
        
        return(int(red), int(green), int(blue))

    def hex_add_alpha(self, value, alpha):
        r, g, b = self.hex_to_rgb(value)
        r, g, b = self.rgba_to_rgb(r, g, b, alpha)
        return(self.rgb_to_hex(r, g, b))



def bezier(p1, p2, control, num=200):
    """ Construct a quadratic Bezier curve. 
    
    Input:
      p1: X,Y coordinates of the starting point.
      
      p2: X,Y coordinates of the end point.
      
      control: X,Y coordinates of the control point.
      
      num: Number of points that should be computed along the curve from p1 to p2.
    
    Returns:
      Array of shape (num,2) holding the X,Y coordinates of the Bezier curve.
    
    """
    T = np.linspace(0, 1, num=num)
    
    # Matrix multiplication is used to compute the coordinates. Information on
    # how to do the computation is derived from here: https://pomax.github.io/bezierinfo/
    
    # Define the initial matrices
    A = np.array([ [1,t,t**2] for t in T ])
    B = np.array([[1,0,0], [-2,2,0], [1,-2,1]])
    C = np.array([[p1[0], control[0], p2[0]], [p1[1], control[1], p2[1]]])
    
    # Compute the matrix product, essentially A*B*C while respecting the correct axes
    E = np.tensordot(B, C, axes=([-1],[1]))
    D = np.tensordot(A, E, axes=([1], [0]))
    
    return(D)


def bezier_ribbons(rads, dist=0.3):
    """ Compute the Bezier ribbons for a chord diagram.
    
    The control point will be set to lie between the two points specified in rads
    at a distance of dist (default 0.3) from the origin.
    
    Input:
      rads: Tuple containing the angle (in radians) of the two points on the
            unit circle that should be connected with a bezier curve.
    
      dist: Distance from the origin for the control point.
    
    Returns:
      The X and Y coordinates of the Bezier curve.
    """
    
    # Convert radians to cartesian coordinates 
    radToXY = lambda rad, r: (r*np.cos(rad), r*np.sin(rad))
    
    # Define the control point for the Bezier curve
    p1 = rads[0]
    p2 = rads[1]
    cp = min(p1,p2) + np.abs(p2 - p1)/2.
    
    # Compute the Bezier curve
    b = bezier(radToXY(p1, 1), radToXY(p2, 1), radToXY(cp, dist))
    
    X, Y = b[:,0], b[:,1]
    
    return(X, Y)



class ChordDiagramDataMapper():
    """ Helper class for the Chord Diagram.
    
    Compute the angles of the ideogram, i.e. of the "outer ring". These are
    used to generate the ideogram itself, as well as the bezier curves connecting
    the ideograms.
    
    arcAngles is a dictionary containing the group (one row of the data) as key,
    and another dictionary as value. The value dictionary contains as keys the
    identifiers of the respective other groups and values the starting and ending
    angles of the segment on the circle. For each pair, the "flow" between the 
    two groups can thus be mapped.
    
    """
    
    def __init__(self, data):
        
        assert( data.shape[0] == data.shape[1] )
        
        self.data = data
        self.C = Color()
        
        # Create a label for each group in the data
        self.groups = np.arange(data.shape[0])
        
        # Assign a color to each group
        self.color = { g:self.C() for g in self.groups }
        
        # Compute the angles (on the unit circle) for the ring representation
        self.arcAngles = self._ideogramBounds()
    
    def getColor(self, group):
        """ Obtain the color for a group. """
        return(self.color[group])
    
    def getArc(self, g, k):
        """
        Input:
          g: The identifier of the first group
          k: The identifier of the second group
        
        Returns:
          Tuple (start angle, end angle) for the two groups.
        """
        return(self.arcAngles[g][k])

    def ideograms(self):
        """ Get the ideogram angle bounds for plotting """
        for g in self.groups:
            yield(self.arcAngles[g], self.color[g])

    
    def _mapRadians(self, x, a, b, gap):
        """ Map value x in [a,b] to radians """
        # http://stackoverflow.com/a/5732390/1922650
        assert( a <= x <= b)
        slope  = (2*np.pi - gap) / (b - a)
        output = slope * (x - a)
        return(output)

        
    def _ideogramBounds(self):
        """ Create line segments that will form the outer ring. """
        gap      = 0.005 * 2*np.pi  # define the size of the gap between the elements
        gapTotal = gap * len(self.groups) # one gap between each group
        
        # Compute the total length needed
        N = self.data.sum()
        
        arcAngles = {  g:dict() for g in self.groups }
        arcAngles[-1] = {'first_position':(0,-gap), }
        
        arcMax = lambda x: max(x, key=itemgetter(1))[1]
        
        for g in self.groups:
            ideogramParts = sorted(self.data[g,:], reverse=True) # high to low
            
            for s, val in enumerate(ideogramParts):
                
                if s == 0: # first entry, add the gap between the ideograms
                    start_angle = arcMax(arcAngles[g-1].values()) + gap
                else:
                    start_angle = arcMax(arcAngles[g].values())
                
                end_angle = start_angle + self._mapRadians(val, 0, N, gapTotal)

                idx = np.where(self.data[g,:] == val)[0][0] # keep i,j indexing for the data matrix
                arcAngles[g][idx] = (start_angle, end_angle)
        
        del arcAngles[-1]
        return(arcAngles)



class ChordDiagram():
    """
    Create a chord diagram for Bokeh.
    
    Provide basic functionality upon which one can build to add required 
    functionality for a use in Bokeh.
    
    The input data is taken to be of the same form as the input in the Plotly
    example: https://plot.ly/python/filled-chord-diagram/
    
    The "interactions" between groups are represented by a square matrix, where
    each row represents one group, and each columns contains the number that is
    "send" from group A to the respective other group. The diagonals are 
    representing the "self" interaction of each group. The matrix can be
    symmetric if no directionality between the groups is present.
    
    """
    
    def __init__(self, data, labels=None):
        """
        Input:
          data:   Square matrix containing the interaction between each group
          
          labels: Not implemented (should be labels for each group)
        """
        
        if data.shape[0] != data.shape[1]:
            raise ValueError('The input data must be a square matrix.')
        
        self.data = data
        self.labels = labels

        # Create group identifiers
        self.groups = np.arange(data.shape[0])
        
        # Compute the angles on the unit circle for each "interaction" between the groups.
        self.chordDiagramDataMapper = ChordDiagramDataMapper(data)

        # Keeping a reference to the figure will allow adding stuff to it later (if needed)
        self.fig = None
        
    def _plotIdeograms(self, line_width=10):
        """ Plot the ideograms (outer rings) """
        ideo = self.chordDiagramDataMapper.ideograms()
        
        for ideo, color in self.chordDiagramDataMapper.ideograms():
            # The values contain the angles for all interactions. The ideogram
            # must start from the smallest value and extent to the largest value.
            ideoStart = min(ideo.values(), key=itemgetter(0))[0]
            ideoStop  = max(ideo.values(), key=itemgetter(0))[1]
            
            # Creat sample points on the line. If circle appears to be too low
            # resolution, increase the num parameter.
            rads  = np.linspace(ideoStart, ideoStop, num=100, endpoint=True)

            # Compute x and y values in carteasian coordinates.
            X, Y = [ np.cos(r) for r in rads ], [ np.sin(r) for r in rads ]
            
            # Map the color of the group to be used with Bokeh
            colorIdeo = self.chordDiagramDataMapper.C.hex_add_alpha(color, 1)
            
            # Plot the line
            self.fig.line(X, Y, color=colorIdeo, alpha=1, line_width=line_width)
       
        return(self.fig)
    
    
    def _bezierPatchSelf(self, g):
        color = self.chordDiagramDataMapper.getColor(g)
        arc   = self.chordDiagramDataMapper.getArc(g, g)
        
        X, Y = None, None
        if arc[0] != arc[1]: # skip "empty"interactions
            X, Y  = self._createBezierPatches(arc)
    
        return(X, Y, color)
    
    
    def _plotBezierUnique(self, group=None, alpha=0.8):
        """ Create a patch for "self" interactions. """
        patchX, patchY, patchColor = list(), list(), list()
        
        plotBeziers = self.groups
        if group is not None:
            plotBeziers = [group, ]
        
        for g in plotBeziers:
            X, Y, color = self._bezierPatchSelf(g)
            
            if X is None: # skip "empty" interactions
                continue
            
            patchX.append(X)
            patchY.append(Y)
            patchColor.append(color)
            

        self.fig.patches(patchX, patchY, color=patchColor, alpha=alpha, level='underlay')
        
        return(self.fig)
        
    def _plotBezierShared(self, g, alpha=0.3):
        """ Create a patch for intergroup interactions.
        
        The parameter g specifies for which group the interactions should be
        shown. The interactions for one group (with all the other groups) can
        be shown at times. This constrained is to maintain the readibility and
        to allow the colors of the interactions to be matched unambigiously.
        """
        N = len(self.groups)

        for j in range(N):
            arc_1 = self.chordDiagramDataMapper.getArc(g, j)
            arc_2 = self.chordDiagramDataMapper.getArc(j, g)
        
            X, Y = self._createBezierPatches(arc_1, arc_2)
            
            color = self.chordDiagramDataMapper.getColor(j)
            
            self.fig.patch(X, Y, fill_color=color, fill_alpha=alpha, 
                           line_color=None, level='underlay')
        
        return(self.fig)
    
        
    def plot(self, group=None):
        """ Create a Chord Diagram.
        
        The group parameter can be the index of the groups (row) for which the
        interactions should be shown. If None, then only the "self" interactions,
        i.e. diagonal elements, are displayed.
        
        Each call creates a new figure which is returned and can be accesses
        through the class itself for later reference.
        
        """
        self.fig = self._create_figure()
        self._plotIdeograms()
        
        if group is None:
            self._plotBezierUnique()
        else:
            self._plotBezierUnique(group)
            self._plotBezierShared(group)

        return(self.fig)
    
    def _create_figure(self, size=500, margin=0.05):
        """ Create a square figure on which the diagram should be drawn. """
        fig = figure(plot_width=size, plot_height=size, toolbar_location=None, tools='tap')
        #fig = figure(plot_width=size, plot_height=size, tools='tap')
        fig.axis.visible = False
        fig.grid.visible = False
        
        plt_min = -1 - margin
        plt_max = +1 + margin
        
        fig.x_range = Range1d(plt_min, plt_max)
        fig.y_range = Range1d(plt_min, plt_max)
        return(fig)

    def _createBezierPatches(self, *args):
        if len(args) == 1:
            X, Y = self._createBezierPatchesUnique(args[0])
        elif len(args) == 2:
            X, Y = self._createBezierPatchesShared(args[0], args[1])
        else:
            raise ValueError("_createBezierPatches() takes one or two arguments only.")
        return(X, Y)
    
    def _createBezierPatchesShared(self, arcA, arcB):
        """ Create the patch starting from the end of the first arc. """
        X, Y = list(), list()
        
        # Add points for the first bezier curve
        x, y = bezier_ribbons((arcA[1], arcB[0]), False)
        X.extend(x)
        Y.extend(y)
        
        # Add points belonging to the ideogram
        rads  = np.linspace(arcB[0], arcB[1], num=50)
        x, y = np.cos(rads), np.sin(rads)
        X.extend(x)
        Y.extend(y)

        # Add points for the second bezier curve
        x, y = bezier_ribbons((arcB[1], arcA[0]), False)
        X.extend(x)
        Y.extend(y)

        # Add points belonging to the ideogram
        rads  = np.linspace(arcA[0], arcA[1], num=50)
        x, y = np.cos(rads), np.sin(rads)
        X.extend(x)
        Y.extend(y)
        
        return(X, Y)


    def _createBezierPatchesUnique(self, arc):
        """ Create the patch that bends back to the same group. """
        X, Y = list(), list()
        
        # Add points for the ribbon
        x, y = bezier_ribbons([arc[1], arc[0]], True)
        X.extend(x)
        Y.extend(y)

        # Add points belonging to the ideogram
        rads  = np.linspace(arc[0], arc[1], num=50)
        x, y = np.cos(rads), np.sin(rads)
        X.extend(x)
        Y.extend(y)

        return(X, Y)
