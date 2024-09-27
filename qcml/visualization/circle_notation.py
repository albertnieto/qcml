# Copyright 2024 Albert Nieto

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
import numpy as np


class CircleNotation:
    """
    CircleNotation class for creating annotated circles on a matplotlib Axes.

    Parameters:
    - label (list): List of labels for each circle.
    - inner_color (list): List of colors for the inner circles.
    - outer_color (list): List of colors for the outer circles.
    - angle (list): List of angles for drawing lines.
    - x_offset (list): List of x-coordinates for circle centers.
    - y_offset (list): List of y-coordinates for circle centers.
    - len_outer_circle (list): List of radii for outer circles.
    - len_inner_circle (list): List of radii for inner circles.
    - line_length (float, optional): Length of lines drawn from circles. Default is DEFAULT_LINE_LENGTH.
    - line_color (str, optional): Color of the lines. Default is DEFAULT_LINE_COLOR.
    - omit_line (list, optional): List of booleans indicating whether to omit lines for each circle. Default is DEFAULT_OMIT_LINE.

    Attributes:
    - DEFAULT_LINE_LENGTH (float): Default length of lines drawn from circles.
    - DEFAULT_LINE_COLOR (str): Default color of the lines.
    - DEFAULT_CIRCLE_COLOR (str): Default color of circles.
    - DEFAULT_TEXT_COLOR (str): Default color of text.
    - DEFAULT_HEAD_WIDTH (float): Default width of arrowhead in lines.
    - DEFAULT_HEAD_LENGTH (float): Default length of arrowhead in lines.
    - DEFAULT_FONT_SIZE (int): Default font size of text.
    - DEFAULT_TEXT_DISPLACEMENT (float): Default vertical displacement of text.
    - DEFAULT_OMIT_LINE (bool): Default value indicating whether to omit lines for each circle.
    - DEFAULT_CIRCLE_FILL (bool): Default value indicating whether circles should be filled.
    - DEFAULT_X_OFFSET (float): Default x-coordinate for circle centers.
    - DEFAULT_Y_OFFSET (float): Default y-coordinate for circle centers.
    - DEFAULT_CIRCLE_RADIUS (float): Default radius of circles.
    - DEFAULT_ANGLE (float): Default angle for drawing lines.
    - DEFAULT_TEXT_HA (str): Default horizontal alignment of text.
    - DEFAULT_TEXT_VA (str): Default vertical alignment of text.
    """

    DEFAULT_LINE_LENGTH = 1
    DEFAULT_LINE_COLOR = "black"
    DEFAULT_CIRCLE_COLOR = "black"
    DEFAULT_TEXT_COLOR = "black"
    DEFAULT_HEAD_WIDTH = 0
    DEFAULT_HEAD_LENGTH = 0
    DEFAULT_FONT_SIZE = 12
    DEFAULT_TEXT_DISPLACEMENT = -1.35
    DEFAULT_OMIT_LINE = False
    DEFAULT_CIRCLE_FILL = False
    DEFAULT_X_OFFSET = 0
    DEFAULT_Y_OFFSET = 0
    DEFAULT_CIRCLE_RADIUS = 1
    DEFAULT_ANGLE = 0
    DEFAULT_TEXT_HA = "center"
    DEFAULT_TEXT_VA = "center"

    def __init__(
        self,
        label,
        inner_color,
        outer_color,
        angle,
        x_offset,
        y_offset,
        len_outer_circle,
        len_inner_circle,
        line_length=DEFAULT_LINE_LENGTH,
        line_color=DEFAULT_LINE_COLOR,
        omit_line=DEFAULT_OMIT_LINE,
    ):
        """
        Initialize CircleNotation instance.

        Parameters:
        - label (list): List of labels for each circle.
        - inner_color (list): List of colors for the inner circles.
        - outer_color (list): List of colors for the outer circles.
        - angle (list): List of angles for drawing lines.
        - x_offset (list): List of x-coordinates for circle centers.
        - y_offset (list): List of y-coordinates for circle centers.
        - len_outer_circle (list): List of radii for outer circles.
        - len_inner_circle (list): List of radii for inner circles.
        - line_length (float, optional): Length of lines drawn from circles. Default is DEFAULT_LINE_LENGTH.
        - line_color (str, optional): Color of the lines. Default is DEFAULT_LINE_COLOR.
        - omit_line (list, optional): List of booleans indicating whether to omit lines for each circle. Default is DEFAULT_OMIT_LINE.
        """

        self.fig, self.ax = plt.subplots()
        self.label = label
        self.inner_color = inner_color
        self.outer_color = outer_color
        self.line_color = line_color
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.len_outer_circle = len_outer_circle
        self.len_inner_circle = len_inner_circle
        self.line_length = line_length
        self.angle = angle
        self.omit_line = omit_line

    def draw_circle(
        self,
        x_offset=DEFAULT_X_OFFSET,
        y_offset=DEFAULT_Y_OFFSET,
        radius=DEFAULT_CIRCLE_RADIUS,
        color=DEFAULT_CIRCLE_COLOR,
        is_filled=DEFAULT_CIRCLE_FILL,
    ):
        """
        Draw a circle on the Axes.

        Parameters:
        - x_offset (float, optional): x-coordinate of the circle center. Default is DEFAULT_X_OFFSET.
        - y_offset (float, optional): y-coordinate of the circle center. Default is DEFAULT_Y_OFFSET.
        - radius (float, optional): Radius of the circle. Default is DEFAULT_CIRCLE_RADIUS.
        - color (str, optional): Color of the circle. Default is DEFAULT_CIRCLE_COLOR.
        - is_filled (bool, optional): Whether the circle should be filled. Default is DEFAULT_CIRCLE_FILL.
        """
        circle = plt.Circle((x_offset, y_offset), radius, color=color, fill=is_filled)
        self.ax.add_patch(circle)

    def draw_text(
        self,
        label,
        x_offset=DEFAULT_X_OFFSET,
        y_offset=DEFAULT_Y_OFFSET,
        text_displacement=DEFAULT_TEXT_DISPLACEMENT,
        color=DEFAULT_TEXT_COLOR,
        fontsize=DEFAULT_FONT_SIZE,
        ha=DEFAULT_TEXT_HA,
        va=DEFAULT_TEXT_VA,
    ):
        """
        Draw text on the Axes.

        Parameters:
        - label (str): Text label to be drawn.
        - x_offset (float, optional): x-coordinate of the text. Default is DEFAULT_X_OFFSET.
        - y_offset (float, optional): y-coordinate of the text. Default is DEFAULT_Y_OFFSET.
        - text_displacement (float, optional): Vertical displacement of the text. Default is DEFAULT_TEXT_DISPLACEMENT.
        - color (str, optional): Color of the text. Default is DEFAULT_TEXT_COLOR.
        - fontsize (int, optional): Font size of the text. Default is DEFAULT_FONT_SIZE.
        - ha (str, optional): Horizontal alignment of the text. Default is DEFAULT_TEXT_HA.
        - va (str, optional): Vertical alignment of the text. Default is DEFAULT_TEXT_VA.
        """
        self.ax.text(
            x_offset,
            y_offset + text_displacement,
            label,
            color=color,
            ha=ha,
            va=va,
            fontsize=fontsize,
        )

    def draw_line(
        self,
        angle=DEFAULT_ANGLE,
        x_offset=DEFAULT_X_OFFSET,
        y_offset=DEFAULT_Y_OFFSET,
        color=DEFAULT_LINE_COLOR,
        head_width=DEFAULT_HEAD_WIDTH,
        head_length=DEFAULT_HEAD_LENGTH,
    ):
        """
        Draw a line on the Axes.

        Parameters:
        - angle (float, optional): Angle of the line. Default is DEFAULT_ANGLE.
        - x_offset (float, optional): x-coordinate of the line start. Default is DEFAULT_X_OFFSET.
        - y_offset (float, optional): y-coordinate of the line start. Default is DEFAULT_Y_OFFSET.
        - color (str, optional): Color of the line. Default is DEFAULT_LINE_COLOR.
        - head_width (float, optional): Width of the arrowhead. Default is DEFAULT_HEAD_WIDTH.
        - head_length (float, optional): Length of the arrowhead. Default is DEFAULT_HEAD_LENGTH.
        """
        line_x = self.line_length * np.cos(angle) - head_length
        line_y = self.line_length * np.sin(angle)

        self.ax.arrow(
            x_offset,
            y_offset,
            line_x,
            line_y,
            head_width=head_width,
            head_length=head_length,
            fc=color,
            ec=color,
        )

    def draw_single(self, i):
        """
        Draw a single set of circles, text, and lines.

        Parameters:
        - i (int): Index of the circle set to be drawn.
        """
        self.draw_circle(
            x_offset=self.x_offset[i],
            y_offset=self.y_offset[i],
            radius=self.len_outer_circle[i],
            color=self.outer_color[i],
        )
        self.draw_circle(
            x_offset=self.x_offset[i],
            y_offset=self.y_offset[i],
            radius=self.len_inner_circle[i],
            color=self.inner_color[i],
            is_filled=True,
        )
        self.draw_text(
            self.label[i], x_offset=self.x_offset[i], y_offset=self.y_offset[i]
        )
        if not self.omit_line[i]:
            self.draw_line(
                angle=self.angle[i],
                x_offset=self.x_offset[i],
                y_offset=self.y_offset[i],
            )

    def draw_all(self):
        """Draw all circle sets."""
        for i in range(len(self.angle)):
            self.draw_single(i)

        # Set aspect ratio to be equal
        self.ax.set_aspect("equal", adjustable="box")

        # Hide axis
        plt.axis("off")

        # Display the plot
        plt.show()
