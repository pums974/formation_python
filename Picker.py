#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class Picker(object):
    """
    Object used to select point on a list

    User may have to modify the dataplot function
    """
    def pick(self, args, ndata=None, plotfx=None):
        """
        Main function
        
        The first dimension of data must be the different images

        Given a dataset, it will
        1. plot the data
        2. wait for the user to close the figure (NOT in Jupyter)
        3. return the picked points
        """
        
        # prepare data
        self.args = args
        if ndata is None:
            if isinstance(data, np.ndarray):
                self.ndata = data.shape[0]
            else:
                self.ndata = len(data)
        else:
            self.ndata = ndata

        if plotfx is None:
            self.plotfx = self.defaultplot
        else:
            self.plotfx = plotfx

        # Current image number
        self.ImgNum = 0

        # prepare result
        self.result = np.empty((self.ndata,2))
        self.result[:] = np.nan

        # prepare figure
        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.ax.set_title('Select the points you want')

        # reduce the ax to give space to the table
        box = self.ax.get_position()
        self.ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        
        # add the two buttons
        # They must be attributes of Picker, otherwise they would be destroyed
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')

        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        
        # declare events
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        self.bprev.on_clicked(self.prev)
        self.bnext.on_clicked(self.next)
        

        # plot data and table
        self.refresh()

        # Everything is ready, wait for the user to pick points (NOT in Jupyter)
        plt.show(block=True)

        # return a view on the result
        return self.result
        
    def refresh(self, onlymarker=False):
        """
        Plot data and print result table

        If onlymarker is true, then we do not redraw the plot
        (hence we keep any zoom the user may have done)
        """
        if onlymarker:
            # Mark the picked point if any
            self.update_marker()
        else:
            # Clear existing plot if any
            self.clearall()
            
            # Plot the data
            self.dataplot(self.args)
            
        # Print the table
        self.tableplot()
                
        # refresh the new plot
        plt.draw()
        
    def clearall(self):
        """
        Clear existing plot if any
        """
        self.ax.cla()
        
    def update_marker(self):
        """
        Mark the picked point if any
        
        we need to store the markers
        """
        self.markers.set_ydata(self.result[self.ImgNum][1])
        self.markers.set_xdata(self.result[self.ImgNum][0])

    @staticmethod
    def defaultplot(ax, ImgNum, args):
        plot, = ax.plot(*args[ImgNum].T, 'ro', picker=50.)
        return plot  

    def dataplot(self, args):
        """
        Plot the data
        
        the picker argument is important to enable picking
        it's value can be ajusted, but a 50 points tolerance seems ok
        
        we need to store the markers and the pickable element

        One can add other plot on the ax 
        """
        self.plot = self.plotfx(self.ax, self.ImgNum, self.args)
        self.markers, = self.ax.plot(*self.result[self.ImgNum], 'go', alpha=0.5, ms=20)

    def tableplot(self):
        """
        Print the table

        we need to store the table, in order to detect where the user clicked
        
        I did not find how to update the table
        """
        # prepare formatters
        data_formatter = np.vectorize(lambda x : "{:5.2f}".format(x) if not np.isnan(x) else "")
        rowlabel_formatter = np.vectorize(lambda x : "{:4d}".format(x))
        color_formatter = np.vectorize(lambda x : "lightgreen".format(x) if len(x) else "lightcoral")

        # create table from data
        colLabels = ("Id", "X", "Y")
        
        cellText = np.zeros((self.ndata, 3), dtype=object)
        cellText[:,1:] = data_formatter(self.result)
        cellText[:, 0] = rowlabel_formatter(range(self.ndata))

        # colorize table
        colors = color_formatter(cellText)
        colors[:,0] = "w"
        colors[self.ImgNum] = "skyblue"

        # print table
        self.table = self.ax.table(cellText=cellText,
                                   colLabels=colLabels,
                                   cellColours=colors,
                                   picker=5,
                                   bbox=(1.05, 0.05,.4,.95))
         
        # adjust width and line style
        for cell in self.table.get_celld().values():
            cell.set_width(.3)
            cell.set_linewidth(0.1)
    
    def onpick(self, event):
        """
        Act when the user clicked
        """
        
        # Ignore non left click
        if event.mouseevent.button != 1:
            return

        # If the user clicked on the plot
        if event.artist is self.plot:
            self.pick_on_plot(event)

        # If the user clicked on the table
        elif event.artist is self.table:
            self.pick_on_table(event)

        # otherwise
        else:
            return

    def pick_on_plot(self,event):
        """
        Select the closest point to the user click and store it in result
        """

        # mouse position
        xmous = event.mouseevent.xdata
        ymous = event.mouseevent.ydata
        if (xmous is None or
            ymous is None) :
            return

        # data position
        xdata = self.plot.get_xdata()
        ydata = self.plot.get_ydata()

        # All points on the 50 points tolerance
        inds = event.ind

        # We need to renormalize the positions
        xmin, xmax = xdata.min(), xdata.max()
        ymin, ymax = ydata.min(), ydata.max()
        xfac = 1./(xmax-xmin)**2
        yfac = 1./(ymax-ymin)**1

        # compute the distances of all the points to the clicked position
        dists = xfac*(xmous - xdata[inds])**2 + yfac*(ymous - ydata[inds]) ** 2

        # the point closest to the mouse
        closest = inds[np.argmin(dists)]

        # Save it
        self.result[self.ImgNum]= [xdata[closest], ydata[closest]]

        # update plot for marker
        self.refresh(onlymarker=True)

    def pick_on_table(self,event):
        """
        Change image to the one the user selected
        
        It is not easy to go from mouse position to cell coordinate !
        """

        # mouse position in the figure coords
        ymous = event.mouseevent.y

        # table position in the figure coords
        ((x0, y0), (x1, y1)) = self.table.get_window_extent(self.fig).get_points()
        
        # mouse position in the table coords
        ymous = (ymous - y0) / (y1 - y0)

        # clicked row
        j = int(self.ndata - ymous * (self.ndata+1))

        # if the user effectively cliked on a row
        if 0 <= j < self.ndata:
            # select it
            self.ImgNum = j
            # refresh the whole figure
            self.refresh()        


    def next(self, event):
        """
        Select the next image if any
        """
        self.ImgNum = min(self.ImgNum + 1, self.ndata - 1)
        self.refresh()

    def prev(self, event):
        """
        Select the previous image if any
        """
        self.ImgNum = max(self.ImgNum - 1,0)
        self.refresh()


