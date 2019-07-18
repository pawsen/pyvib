#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.pyplot as plt
import numpy as np

class _rfsPlotBuilder(object):
    """
    Could be update to not clear and redraw, but instead just update the
    points.

    First save handle to plot:
    3dplot = ax3d.plot(y,yd,-ydd, '.k', markersize=10)
    ax3d.set_autoscaley_on(True)

    #¤¤ on update:
    3dplot.set_xdata(xdata)
    # Rescale:
    ax3d.relim()
    ax3d.autoscale_view()

    # draw and flush (something like this...)
    fig.canvas.draw()
    fig.canvas.flush_events()

    """
    def __init__(self, rfs):

        self.rfs = rfs


        fig= plt.figure()
        ax = fig.add_subplot(211)
        ax2 = fig.add_subplot(223, projection='3d')
        ax3 = fig.add_subplot(224)
        self.ax3d = ax2
        self.ax2d = ax3
        self.fig = fig
        # fig.subplots_adjust(wspace=0.7)
        # fig.subplots_adjust(hspace=0.5)
        #plt.suptitle('RFS for DOF {}'.format(self.rfs.dofs[0]))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)

        # plot acceleration signal
        t = np.arange(self.rfs.ns)/self.rfs.fs
        ax.plot(t, self.rfs.ydd,'-k')
        #ax.plot(t, self.rfs.y,'-k')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'Acceleration ($m/s^2$)')
        ax.set_title('Acceleration, for DOF {}'.format(self.rfs.dofs[0]))

        #¤ back to rectangular settings
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        xx = [xmin,xmax]
        # Mouse should be within 2% of line.
        self.epsilon = np.diff(xx) * 0.02

        # Set initial selection and add rectangle
        start_sel = 0.45
        width_sel = 0.1

        x1 = xx[0] + start_sel * np.diff(xx)
        dx = width_sel * np.diff(xx)
        rect = patches.Rectangle((x1, ymin), dx, ymax-ymin, alpha=0.4,fc='y')
        ax.add_patch(rect)

        # ax position rect [left, bottom, width, height] in fractions of figure
        # width and height.
        # checkbutton showing stiffness or damping
        buttonax = fig.add_axes([0.8, 0.02, 0.15, 0.06]) #, fc='white')
        button = CheckButtons(buttonax, ('Damping',), (False,))
        button.on_clicked(self.checkbox_clicked)

        # set slider
        sliderax = fig.add_axes([0.1, 0.02, 0.6, 0.03], fc='white')
        slider = Slider(sliderax, 'Tol', 0, 1.0, valinit= 0.05)
        slider.on_changed(self.slider_update)
        slider.drawon = True

        canvas = rect.figure.canvas
        self.canvas = canvas
        self.rect = rect
        self.axes = rect.axes
        self.ind = None
        self.show_damped = False

        # show rfs for the initial selection
        self.update_rfs()

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

        plt.show()


    def slider_update(self, value):

        self.rfs.tol_slice = value
        # y, yd, ydd, y_tol, yd_tol, ydd_tol = self.rfs.update_sel(self.idx1, self.idx2)
        self.update_rfs()

        # TODO: works, even if the canvas belong to rect. ie canvas = rect.canvas
        self.canvas.blit(self.ax2d.bbox)
        self.canvas.blit(self.ax3d.bbox)

    def get_ind(self, event):
        # Figure out which line to move, or move the figure if in between lines.

        # get the index of the vertex under point if within epsilon tolerance
        x, y = self.rect.get_x(), self.rect.get_y()
        width = self.rect.get_width()
        # d1: distance to left line, d2: distance to right line
        d1 = np.sqrt((x-event.xdata)**2)
        d2 = np.sqrt(((x+width)-event.xdata)**2)

        # save cursor position in order to move the rectangle a relative
        # distance
        self.press = event.xdata, x, width

        epsilon = self.epsilon
        # only move if cursor is sufficiently close to line or over rect
        if d1 < epsilon:
            return 'left'
        elif d2 < epsilon:
            return 'right'
        elif event.xdata > x and event.xdata < x+width:
            return 'move'
        else:
            return None

    def check_rect(self, event):
        # only do stuff if we're on the rectangle
        if event.inaxes != self.rect.axes:
            return False
        elif event.button != 1:
            return False
        else:
            return True

    def button_press_callback(self, event):
        if self.check_rect(event) is False: return

        # get line cursor is close to
        self.ind = self.get_ind(event)

        self.rect.set_animated(True)
        self.canvas.draw()
        # store background for updating the plot
        self.background = self.canvas.copy_from_bbox(self.rect.axes.bbox)
        self.axes.draw_artist(self.rect)
        self.canvas.blit(self.axes.bbox)

    def button_release_callback(self, event):
        if self.check_rect(event) is False: return
        self.ind = None
        # update graphs showing restoring force
        self.update_rfs()

        self.rect.set_animated(False)
        self.background = None
        self.rect.figure.canvas.draw()

    def motion_notify_callback(self, event):
        # Update the rect position while dragging/moving
        if self.check_rect(event) is False: return
        if self.ind is None:
            return

        # get coords from when the mouse was pressed
        x_event, x_rect, w_rect = self.press
        x, y = event.xdata, event.ydata
        dx = x - x_event
        if self.ind == 'left':
            self.rect.set_x(x)
            self.rect.set_width(w_rect - dx)
        elif self.ind == 'right':
            self.rect.set_width(x-x_rect)
        else:  #move
            self.rect.set_x(x_rect + dx)

        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.rect)
        self.canvas.blit(self.axes.bbox)

    def checkbox_clicked(self, label):
        # Flip value
        self.show_damped = not self.show_damped
        self.update_rfs()
        self.rect.figure.canvas.draw()


    def set_postion(self, x, width):
        self.rect.set_x(x)
        self.rect.set_width(width)
        self.update_rfs()

    def update_rfs(self):
        # plot RFS

        show_damped = self.show_damped
        # get index of selection
        x1, width = self.rect.get_x(), self.rect.get_width()
        x2 = x1 + width
        xmin, xmax = self.axes.get_xlim()
        xx = [xmin, xmax]
        self.idx1 = int(np.floor( self.rfs.ns * (x1 - xx[0]) / np.diff(xx)))
        self.idx2 = int(np.ceil( self.rfs.ns * (x2 - xx[0]) / np.diff(xx)))

        y, yd, ydd, y_tol, yd_tol, ydd_tol = self.rfs.update_sel(self.idx1,
                                                                 self.idx2,
                                                                 show_damped)
        self.ax3d.clear()
        self.ax3d.plot(y,yd,-ydd, '.k', markersize=2)
        self.ax3d.plot(y_tol,yd_tol,-ydd_tol, '.r', markersize=10)
        #self.ax3d.set_zlim([-5,5])

        self.ax3d.set_title("Restoring force surface")
        self.ax3d.set_xlabel('Displacement (m)')
        self.ax3d.set_ylabel('Velocity (m/s)')
        self.ax3d.set_zlabel('-Acceleration (m/s²)')
        self.ax3d.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        self.ax2d.clear()
        if show_damped:
            self.ax2d.plot(yd_tol,-ydd_tol, '.k', markersize=8)
            self.ax2d.set_title('Damping curve')
            self.ax2d.set_xlabel('Rel. velocity (m/s)')
        else:
            self.ax2d.plot(y_tol,-ydd_tol, '.k', markersize=8)
            self.ax2d.set_title('Stiffness curve')
            self.ax2d.set_xlabel('Rel. displacement (m)')
        self.ax2d.axhline(y=0, ls='--', lw='0.5',color='k')
        self.ax2d.axvline(x=0, ls='--', lw='0.5',color='k')
        self.ax2d.set_ylabel('-Acceleration (m/s²)')
        self.ax2d.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


from .signal import Signal
class RFS(Signal):
    def __init__(self, signal, dof=0, show_damped = False):
        """ Show a Restoring Force Surface, which gives a visual idea of the
        type of nonlinearity(if any)

        Simply found as
         # g( x_i - x_j, dx_i - dx_j) ~ -ddx_i


        Parameters:
        -----------
        dof : list: [dof_i], [dof_i, dof_j] or empty
            dofs to compare. If none, then compare to ground. If [dof_i], then
            compare dof_i to ground. Else compare dof_i to dof_j
        show_damp : bool
            show stifness or damping coeff.
        """
        dof = np.atleast_1d(dof)
        # if not isinstance(dofs, list):
        #     raise TypeError('dofs not a list. Should be [dof1, dof2]', type(dofs))
        if len(dof.shape) > 2:
            raise ValueError('List of dof is too long. Max two dofs', len(dof))

        self.dofs = dof
        self.ns = signal.ns
        self.fs = signal.fs
        self.tol_slice = 1e-2
        self.signal = signal

        y = signal.y  #[dofs,:]
        yd = signal.yd  #[dofs,:]
        if len(dof) == 1:
            # connected to ground
            self.y = y[dof[0],:]  #.squeeze()
            self.yd = yd[dof[0],:]  #.squeeze()
        else:
            # connected to another dof
            self.y = y[dof[0],:] - y[dof[1],:]
            self.yd = yd[dof[0],:] - yd[dof[1],:]

        self.ydd = signal.ydd[dof[0],:]





    def update_sel(self, id0, id1=-1, show_damped=False):
        """ Update RFS for the given selection

        Parameters:
        ----------
        id0/id1 : int
            index start/end for the selection
        """
        #t = self.t( id0:di1 );
        y = self.y[id0:id1]
        yd = self.yd[id0:id1]
        ydd = self.ydd[id0:id1]

        if show_damped:
            tol =  self.tol_slice * max( np.abs(y))
            ind_tol = np.where(np.abs(y) < tol)
        else:
            tol =  self.tol_slice * max( np.abs(yd))
            ind_tol = np.where(np.abs(yd) < tol)
        y_tol = y[ind_tol]
        yd_tol = yd[ind_tol]
        ydd_tol = ydd[ind_tol]

        return y, yd, ydd, y_tol, yd_tol, ydd_tol

    def plot(self, show_damped=False):
        self.plot = _rfsPlotBuilder(self)


# from mpl_toolkits.mplot3d import Axes3D

# plt.figure(2)
# plt.clf()
# ax = plt.axes(projection='3d')
# ax.plot(y1, yd, -ydd, '.k', markersize=10)
# ax.plot(y1_tol, y1_tol, -ydd_tol, '.r', markersize=12)
# ax.set_title("Restoring force surface")
# ax.set_xlabel('Displacement (m)')
# ax.set_ylabel('Velocity (m/s)')
# ax.set_zlabel('-Acceleration (m/s²)')


# plt.figure(3)
# plt.clf()
# plt.title('Stiffness curve')
# plt.xlabel('Displacement (m)')
# plt.ylabel('-Acceleration (m/s²)')
# plt.plot(y1_tol,-ydd_tol,'.k', markersize=12)

# #plt.show()
