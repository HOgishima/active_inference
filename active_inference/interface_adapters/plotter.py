import numpy as np
import xarray as xr
import hvplot.xarray
import holoviews as hv
from holoviews import opts
from active_inference.interface_adapters.stastics import block_mean, block_stderr
# Jupyter環境での初期化
from bokeh.io import output_notebook
output_notebook()

# probability plot
#-------------------------------------------------
def mean_probability_plot(block_means: xr.DataArray) -> hv.Overlay:
    """Create a plot with lines and confidence intervals for 2D data"""
    
    plot_variable: str = block_means.dims[-1]
    plot_component: str = str(block_means.coords[plot_variable].values)

    if len(block_means.shape) > 1:
        line_plots: hv.Overlay = block_means.hvplot.line(
            by = plot_variable,
            label = plot_component,
        )
    else:
        line_plots: hv.Overlay = block_means.hvplot.line(
            label = plot_component,
        )
    
    return line_plots

def mean_probability_plot_with_stderrs(block_means: xr.DataArray, block_stderrs: xr.DataArray) -> hv.Overlay:
    """Create a plot with lines and confidence intervals for 2D data"""

    mean_plots: hv.Overlay = mean_probability_plot(block_means)     

    time_steps: np.ndarray = block_means[block_means.dims[0]].values
    
    area_plots_list: list[hv.Area] = []

    if len(block_means.shape) > 1:
        plot_variable: str = block_means.dims[-1]
        plot_component: list[str] = list(block_means.coords[plot_variable].values)
    
        for i, name in enumerate(plot_component):
            area_plots = hv.Area(
                (time_steps,  # x_axis
                block_means.isel({plot_variable: i}) - block_stderrs.isel({plot_variable: i}),  # lower_bound
                block_means.isel({plot_variable: i}) + block_stderrs.isel({plot_variable: i}),  # upper_bound
                ),  # upper_bound
                vdims = ['lower_bound', 'upper_bound']
            ).opts(alpha=0.2, line_alpha=0)
            area_plots_list.append(area_plots)
    else:
        area_plots = hv.Area(
            (time_steps,  # x_axis
            block_means - block_stderrs,  # lower_bound
            block_means + block_stderrs,  # upper_bound
            ),  # upper_bound
            vdims = ['lower_bound', 'upper_bound']
        ).opts(alpha=0.2, line_alpha=0)

        area_plots_list.append(area_plots)

    return mean_plots * hv.Overlay(area_plots_list)

def create_probability_plot(
    means: xr.DataArray,
    stderrs: xr.DataArray, 
    width: int = 800, height: int = 300
) -> hv.Overlay:
    """main line plot"""
    # label of line
    time_steps: np.ndarray = np.arange(means.shape[0])
    line_labels: np.ndarray = means.coords[means.dims[-1]].values
    
    # カラーマップの定義
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    plots: list[hv.Overlay] = []
    for i, state in enumerate(line_labels):
        #color = colors[i % len(colors)]

        # main line
        if len(means.dims) > 0:
            variable_label: str = means.dims[-1]
        else:
            variable_label: str = means.indexs.keys()[0]

        print(variable_label)
        line = means.isel({variable_label: i}).hvplot(label=str(state), line_width=2)
        
        # confidence interval
        confidence = hv.Area(
            (
                time_steps,  # x
                means.isel({variable_label: i}) - stderrs.isel({variable_label: i}),  # y
                means.isel({variable_label: i}) + stderrs.isel({variable_label: i})   # y2
            ),
            vdims=['lower_bound', 'upper_bound']  # vdimsをキーワード引数として渡す
        ).opts(alpha=0.2, line_alpha=0)
        
        plots.extend([line, confidence])
    
    return hv.Overlay(plots)

def create_phase_markers(
    baseline_step: int,
    sound_step: int,
    y_pos: float,
    data_length: int
) -> tuple[hv.Overlay, hv.Overlay]:
    """vertical line and label"""

    # vertical line
    vlines = hv.Overlay([
        hv.VLine(baseline_step).opts(color='gray', line_dash='dashed'),
        hv.VLine(sound_step).opts(color='gray', line_dash='dashed')
    ])
    
    # text label
    labels = hv.Overlay([
        hv.Text(baseline_step/2, y_pos, 'Baseline').opts(text_font_size='10pt'),
        hv.Text((baseline_step + sound_step)/2, y_pos, 'Sound').opts(text_font_size='10pt'),
        hv.Text((sound_step + data_length)/2, y_pos, 'No Shock').opts(text_font_size='10pt')
    ])
    
    return vlines, labels

def probability_plot(
    dataset: xr.Dataset,
    baseline_step: int = 80,
    sound_step: int = 120
) -> hv.Layout:
    
    """main plot"""
    # calculate mean and stderr
    means = block_mean(dataset)
    stderrs = block_stderr(dataset)
    
    # create components
    main_plot = mean_probability_plot_with_stderrs(means, stderrs)
    
    # vertical line and label
    vlines, labels = create_phase_markers(
        baseline_step=baseline_step,
        sound_step=sound_step,
        y_pos=float(means.max()),
        data_length=means.shape[0]
    )
    
    # combine and set plot
    plot = (main_plot * vlines * labels).opts(
        width=800,
        height=300,
        legend_position='right',
        show_grid=True,
        title= f"Probability of {means.name}"
    )
    
    return plot