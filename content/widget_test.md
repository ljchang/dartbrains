---
title: Widget Test
---

# Anywidget Test

Testing the compass widget with the MyST `{anywidget}` directive.

## Compass Widget

:::{anywidget} /Code/js/compass_widget.js
{
  "b0": 3.0
}
:::

## Net Magnetization Widget

:::{anywidget} /Code/js/net_magnetization_widget.js
{
  "n_protons": 100,
  "b0_on": false
}
:::

## Precession Widget

:::{anywidget} /Code/js/precession_widget.js
{
  "b0": 3.0,
  "flip_angle": 90.0,
  "t1": 0.0,
  "t2": 0.0,
  "show_relaxation": false,
  "paused": false
}
:::
