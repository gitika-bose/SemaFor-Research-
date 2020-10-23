"""This script is used to generate
a visualization based on the attention
layer output at data_pipeline/analysis
"""
from typing import Tuple, Text


def color_interpolation(rgb_1: Tuple, rgb_2: Tuple, ratio: float)\
        -> Tuple:
    """This function is used to interpolate color between two
    different endpoint. Using color to generate data.

    :rgb_{} --- color representation starting from 0 - 255
    """
    assert len(rgb_1) == len(rgb_2)
    assert 0 <= ratio <= 1

    # start to interpolate rgba
    new_rgb = []
    for dim_a, dim_b in zip(rgb_1, rgb_2):
        new_dim = dim_a + ratio * (dim_b - dim_a)
        new_rgb.append(new_dim)

    return tuple(new_rgb)


def alpha_interpolation(rgba: Tuple, ratio: float):
    """This function interpolate with the alpha parameter
    """
    assert len(rgba) == 4
    assert 0 <= ratio <= 1
    return (rgba[0], rgba[1], rgba[2], rgba[3] * ratio)


def c_str(color_tuple: Tuple):
    """Transfer color_tuple into string repr
    """
    c_string = ', '.join([str(dim) for dim in color_tuple])
    if len(color_tuple) == 3:
        return 'rgb(' + c_string + ')'
    else:
        return 'rgba(' + c_string + ')'


def generate_html(data_path: Text, targ_path: Text):
    """Read the data_path input and draw
    some targ_path output. for html

    :data_path --- path for the attention data, every line contains "TOKEN HEAD_1_PROB HEAD_2_PROB ... HEAD_M_PROB"
    :targ_path --- path for the generated attn html
    """
    attn_color = (106, 90, 205, .6)

    tokens = []
    data_point = []

    html_file = open(targ_path, 'w', encoding='utf-8')
    html_file.write('<body><div>')
    with open(data_path, 'r', encoding='utf-8') as data_file:
        for idx, line in enumerate(data_file):
            #  if idx == 0:
            #      assert line.startswith('prob:')
            #      prob = float(line.strip().split(':')[1])
            #      color = color_interpolation(prob_zero_color,
            #                                  prob_one_color, prob)
            #      html_file.write('<p> Predicted prob: <span style="color: '
            #                      + c_str(color) + ';">' + str(prob)
            #                      + '</span></p>')
            #      continue
            #  elif idx == 2:
            #      assert line.startswith('label:')
            #      label = line.strip().split(':')[1]
            #      html_file.write('<p>Label: ' + label + '</p>\n')
            #      continue
            #
            #  elif not line.strip():
            #      continue
            if not line.strip():
                continue  # previously the script is run on more complex data structure.

            # now we read data from file
            items = line.strip().split()
            tokens.append(items[0])
            data_point.append([float(d) for d in items[1:]])

    # try to normalize data_point
    normalizer = max([max(line) for line in data_point])

    # now  we can start the table plot
    html_file.write('<table>\n')

    for tk, prob_list in zip(tokens, data_point):
        html_file.write('<tr>')
        html_file.write('<td>' + tk + '</td>')
        for p in prob_list:
            rgba = color_interpolation((0, 0, 0, 0),
                                       attn_color, p / normalizer)
            html_file.write('<td style="color: grey; background-color: ' +
                            c_str(rgba) + '";>' + str(p) + '</td>')

        html_file.write('</tr>')

    html_file.write('</table>\n')

    html_file.write('</div></body>')
    html_file.close()


if __name__ == '__main__':
    generate_html('../../data/data_pipeline/analysis/sent_0_bt_0.txt',
                  '../../data/data_pipeline/html_vis/sent_0_bt_0.html')
