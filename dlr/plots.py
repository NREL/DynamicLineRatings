import numpy as np
import pandas as pd
import cmocean
import matplotlib.pyplot as plt
import matplotlib as mpl


############################
### General plotting helpers

def plotparams():
    """Format plots"""
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['mathtext.rm'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 'x-large'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = 'large'
    plt.rcParams['ytick.labelsize'] = 'large'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['figure.figsize'] = 5.0, 3.75
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2.5
    plt.rcParams['ytick.minor.size'] = 2.5


def rainbowmapper(iterable, colormap=None, explicitcolors=False, categorical=False):
    categoricalrainbow = [
        plt.cm.tab20(i) for i in [10,11,6,7,2,3,12,13,16,17,4,5,18,19,0,1,8,9,14,15,]
    ]
    if colormap is not None:
        if isinstance(colormap, list):
            colors=[colormap[i] for i in range(len(iterable))]
        else:
            colors=[colormap(i) for i in np.linspace(0,1,len(iterable))]
    elif len(iterable) == 1:
        colors=['C3']
    elif len(iterable) == 2:
        colors=['C3','C0']
    elif len(iterable) == 3:
        colors=['C3','C2','C0']
    elif len(iterable) == 4:
        colors=['C3','C1','C2','C0']
    elif len(iterable) == 5:
        colors=['C3','C1','C2','C0','C4']
    elif len(iterable) == 6:
        colors=['C5','C3','C1','C2','C0','C4']
    elif len(iterable) == 7:
        colors=['C5','C3','C1','C8','C2','C0','C4']
    elif len(iterable) == 8:
        colors=['C5','C3','C1','C8','C2','C0','C4','k']
    elif len(iterable) == 9:
        colors=['C5','C3','C1','C8','C2','C9','C0','C4','k']
    elif len(iterable) == 10:
        colors=['C5','C3','C1','C6','C8','C2','C9','C0','C4','k']
    elif len(iterable) <= 20:
        colors = categoricalrainbow[:len(iterable)]
    elif categorical:
        colors = categoricalrainbow * (len(iterable)//20 + 1)
    else:
        colors=[plt.cm.rainbow(i) for i in np.linspace(0,1,len(iterable))]
    out = dict(zip(iterable, colors))
    if explicitcolors:
        explicit = {
            'C0': '#1f77b4', 'C1': '#ff7f0e', 'C2': '#2ca02c', 'C3': '#d62728',
            'C4': '#9467bd', 'C5': '#8c564b', 'C6': '#e377c2', 'C7': '#7f7f7f',
            'C8': '#bcbd22', 'C9': '#17becf',
        }
        if len(iterable) <= 10:
            out = {c: explicit[out[c]] for c in iterable}
    return out


def make_cmap(
    vmin=-25,
    vmax=100,
    cmhi=cmocean.cm.rain,
    cmlo=cmocean.cm.amp_r,
    cliphi=10,
    cliplo=10,
    center=0,
):
    cmap_above = cmocean.tools.crop_by_percent(cmhi, cliphi, which='max')
    cmap_below = cmocean.tools.crop_by_percent(cmlo, cliplo, which='min')

    decimals = 3
    vspace = 10**-decimals
    belowsteps = np.arange(vmin, center, vspace)
    abovesteps = np.arange(center, vmax+vspace, vspace)

    colors_above = cmap_above(np.linspace(0,1,len(abovesteps)))
    colors_below = cmap_below(np.linspace(0,1,len(belowsteps)))

    cmap = mpl.colors.ListedColormap(np.vstack((colors_below, colors_above)))

    return cmap


def _despine_sub(ax, 
    top=False, right=False, left=True, bottom=True,
    direction='out'):
    """
    """
    if not top:
        ax.spines['top'].set_visible(False)
    if not right:
        ax.spines['right'].set_visible(False)
    if not left:
        ax.spines['left'].set_visible(False)
    if not bottom:
        ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both',
                   direction=direction, 
                   top=top, right=right, 
                   left=left, bottom=bottom)

def despine(ax=None, 
    top=False, right=False, left=True, bottom=True,
    direction='out'):
    """
    """
    if ax is None:
        ax = plt.gca()
    if type(ax) is np.ndarray:
        for sub in ax:
            if type(sub) is np.ndarray:
                for subsub in sub:
                    _despine_sub(subsub, top, right, left, bottom, direction)
            else:
                _despine_sub(sub, top, right, left, bottom, direction)
    else:
        _despine_sub(ax, top, right, left, bottom, direction)


def get_coordinates(keys, aspect=None, nrows=None, ncols=None):
    """
    Get a grid of subplots from a list of plot keys, with an aspect ratio
    roughly defined by the `aspect` input.

    Outputs
    -------
    tuple: (nrows [int], ncols [int], coords [dict])
    """
    if (not aspect) and (not nrows) and (not ncols):
        aspect = 1.618
    if ncols:
        _ncols = ncols
    else:
        if nrows:
            _ncols = len(keys) // nrows + bool(len(keys) % nrows)
        else:
            _ncols = max(min(int(np.around(np.sqrt(len(keys)) * aspect, 0)), len(keys)), 1)

    if nrows:
        _nrows = nrows
    else:
        _nrows = len(keys) // _ncols + int(bool(len(keys) % _ncols))

    if (_ncols == 1) or (_nrows == 1):
        coords = dict(zip(keys, range(max(_nrows, _ncols))))
    else:
        coords = dict(zip(keys, [(row,col) for row in range(_nrows) for col in range(_ncols)]))

    return _nrows, _ncols, coords


######################
### Geospatial helpers

def get_latlonlabels(df, lat=None, lon=None, columns=None):
    """Try to find latitude and longitude column names in a dataframe"""
    ### Specify candidate column names to look for
    lat_candidates = ['latitude', 'lat']
    lon_candidates = ['longitude', 'lon', 'long']
    if columns is None:
        columns = df.columns

    latlabel = None
    lonlabel = None

    if lat is not None:
        latlabel = lat
    else:
        for col in columns:
            if col.lower().strip() in lat_candidates:
                latlabel = col
                break

    if lon is not None:
        lonlabel = lon
    else:
        for col in columns:
            if col.lower().strip() in lon_candidates:
                lonlabel = col
                break
    
    return latlabel, lonlabel


def df2gdf(dfin, crs='ESRI:102008', lat=None, lon=None):
    """Convert a pandas dataframe with lat/lon columns to a geopandas dataframe of points"""
    ### Imports
    import os
    import geopandas as gpd
    import shapely
    os.environ['PROJ_NETWORK'] = 'OFF'

    ### Convert
    df = dfin.copy()
    latlabel, lonlabel = get_latlonlabels(df, lat=lat, lon=lon)
    df['geometry'] = df.apply(
        lambda row: shapely.geometry.Point(row[lonlabel], row[latlabel]), axis=1)
    df = gpd.GeoDataFrame(df, crs='EPSG:4326').to_crs(crs)

    return df


##########################
### Specific plot elements

def addcolorbarhist(
    f, ax0, data, 
    title=None,
    cmap=plt.cm.viridis, 
    bins=None,
    nbins=201, 
    vmin='default',
    vmax='default',
    cbarleft=1.05,
    cbarwidth=0.025,
    cbarheight=0.5,
    cbarbottom=None,
    cbarhoffset=0,
    histpad=0.005,
    histratio=3,
    labelpad=0.03,
    title_fontsize='large',
    title_alignment=None,
    title_weight='bold',
    ticklabel_fontsize='medium',
    log=False,
    histcolor='0.5',
    extend='neither',
    extendfrac=(0.05,0.05),
    orientation='vertical'):
    """
    Notes
    -----
    * All dimensions are in fraction of major axis size
    * cmap must a colormap object (e.g. plt.cm.viridis)
    * data should be of type np.array
    """
    
    ########################
    ### Imports and warnings
    import matplotlib as mpl
    ### Warnings
    #############
    ### Procedure

    ### Get bounds and make bins
    if vmin == 'default':
        vmin = data.min()
    if vmax == 'default':
        vmax = data.max()
        
    if bins is None:
        bins = np.linspace(vmin, vmax, nbins)
    elif type(bins) is np.ndarray:
        pass
    else:
        print(type(bins), bins)
        print(type(nbins, nbins))
        raise Exception('Specify bins as np.ndarray or nbins as int')
    ax0x0, ax0y0, ax0width, ax0height = ax0.get_position().bounds
    
    ### Defaults for colorbar position
    if (cbarbottom is None) and (orientation == 'horizontal'):
        cbarbottom = 1.05
    elif (cbarbottom is None) and (orientation == 'vertical'):
        cbarbottom = (1 - cbarheight) / 2

    ### If extending the colorbar, clip the ends of the data distribution so
    ### the extended values show up in the histogram
    if extend == 'neither':
        data_hist = data
    elif extend == 'max':
        data_hist = data.clip(upper=vmax)
    elif extend == 'min':
        data_hist = data.clip(lower=vmin)
    elif extend == 'both':
        data_hist = data.clip(lower=vmin, upper=vmax)

    ######### Add colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ### Horizontal orientation
    if orientation in ['horizontal', 'h']:
        caxleft = ax0x0 + ax0width * (1 - cbarheight) / 2 * (1 + cbarhoffset)
        caxbottom = ax0y0 + ax0height * cbarbottom
        caxwidth = cbarheight * ax0width
        caxheight = cbarwidth * ax0height
        
        cax = f.add_axes([caxleft, caxbottom, caxwidth, caxheight])

        _cb1 = mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation='horizontal',
            extend=extend, extendfrac=extendfrac)
        cax.xaxis.set_ticks_position('bottom')

        ##### Add histogram, adjusting for extension length if necessary
        haxleft = caxleft + (
            (extendfrac[0] * caxwidth) if extend in ['min','both']
            else 0
        )
        haxbottom = caxbottom + (cbarwidth + histpad) * ax0height
        if extend == 'neither':
            haxwidth = caxwidth
        elif extend == 'max':
            haxwidth = caxwidth - extendfrac[1] * caxwidth
        elif extend == 'min':
            haxwidth = caxwidth - extendfrac[0] * caxwidth
        elif extend == 'both':
            haxwidth = caxwidth - sum(extendfrac) * caxwidth
        haxheight = histratio * cbarwidth * ax0height

        hax = f.add_axes([haxleft, haxbottom, haxwidth, haxheight])
        ### Plot the histogram
        hax.hist(data_hist, bins=bins, color=histcolor, 
                 log=log, orientation='vertical')
        hax.set_xlim(vmin, vmax)
        hax.axis('off')

        if title is not None:
            if title_alignment is None:
                title_alignment = 'bottom center'

            xy = {
                'bottom': (0.5, -labelpad),
                'top': (0.5, 1+labelpad+histpad+histratio),
                'gap right': (1+labelpad, 1 + histpad/2),
                'gap left': (-labelpad, 1 + histpad/2),
                'both right': (1+labelpad, (1+histpad+histratio)/2),
                'both left': (-labelpad, (1+histpad+histratio)/2),
            }
            xy['bottom center'], xy['top center'] = xy['bottom'], xy['top']
            xy['center bottom'], xy['center top'] = xy['bottom'], xy['top']
            
            va = {
                'bottom': 'top',
                'top': 'bottom',
                'gap right': 'center',
                'gap left': 'center',
                'both right': 'center',
                'both left': 'center',
            }
            va['bottom center'], va['top center'] = va['bottom'], va['top']
            va['center bottom'], va['center top'] = va['bottom'], va['top']
            
            ha = {
                'bottom': 'center',
                'top': 'center',
                'gap right': 'left',
                'gap left': 'right',
                'both right': 'left',
                'both left': 'right'
            }
            ha['bottom center'], ha['top center'] = ha['bottom'], ha['top']
            ha['center bottom'], ha['center top'] = ha['bottom'], ha['top']

            cax.annotate(
                title, fontsize=title_fontsize, weight=title_weight,
                xycoords='axes fraction',
                va=va[title_alignment], xy=xy[title_alignment],
                ha=ha[title_alignment])

        ###### Relabel the first and last values if extending the colorbar
        cax.tick_params(labelsize=ticklabel_fontsize)
        plt.draw()
        xticks = cax.get_xticks()
        xticklabels = [c._text for c in cax.get_xticklabels()]
        if extend == 'neither':
            pass
        if extend in ['min','both']:
            xticklabels[0] = '≤' + xticklabels[0]
        if extend in ['max','both']:
            xticklabels[-1] += '+'
        if extend != 'neither':
            cax.set_xticks(xticks)
            cax.set_xticklabels(xticklabels)
    
    ### Vertical orientation
    elif orientation in ['vertical', 'vert', 'v', None]:
        caxleft = ax0width + ax0x0 + (ax0width * (cbarleft - 1))
        caxbottom = ax0y0 + ax0height * cbarbottom
        caxwidth = cbarwidth * ax0width
        caxheight = cbarheight * ax0height

        cax = f.add_axes([caxleft, caxbottom, caxwidth, caxheight])

        _cb1 = mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation='vertical',
            extend=extend, extendfrac=extendfrac)
        cax.yaxis.set_ticks_position('left')

        ##### Add histogram
        haxleft = caxleft + (cbarwidth + histpad) * ax0width
        haxbottom = caxbottom + (
            extendfrac[0] * caxheight if extend in ['min','both']
            else 0
        )
        haxwidth = histratio * cbarwidth * ax0width
        if extend == 'neither':
            haxheight = caxheight
        elif extend == 'max':
            haxheight = caxheight - extendfrac[1] * caxheight
        elif extend == 'min':
            haxheight = caxheight - extendfrac[0] * caxheight
        elif extend == 'both':
            haxheight = caxheight - sum(extendfrac) * caxheight

        hax = f.add_axes([haxleft, haxbottom, haxwidth, haxheight])

        hax.hist(data_hist, bins=bins, color=histcolor, 
                 log=log, orientation='horizontal')
        hax.set_ylim(vmin, vmax)
        hax.axis('off')

        if title is not None:
            ## 'both center': align to center of cbar + hist
            ## 'cbar center': align to center of cbar
            ## 'cbar left': align to left of cbar
            if title_alignment is None:
                title_alignment = 'gap center'

            xy = {
                'both center': ((1 + histpad + histratio)/2, 1+labelpad),
                'cbar center': (0.5, 1+labelpad),
                'cbar left': (0,1+labelpad),
                'hist right': (histratio + histpad + 1, 1+labelpad),
                'gap center': (1 + histpad/2, 1+labelpad),
            }
            horizontalalignment = {
                'both center': 'center',
                'cbar center': 'center',
                'cbar left': 'left',
                'hist right': 'right',
                'gap center': 'center',
            }

            cax.annotate(
                title, fontsize=title_fontsize, weight=title_weight,
                xycoords='axes fraction',
                verticalalignment='bottom', xy=xy[title_alignment],
                horizontalalignment=horizontalalignment[title_alignment])

        ###### Relabel the first and last values if extending the colorbar
        cax.tick_params(labelsize=ticklabel_fontsize)
        plt.draw()
        yticks = cax.get_yticks()
        yticklabels = [c._text for c in cax.get_yticklabels()]
        if extend == 'neither':
            pass
        if extend in ['min','both']:
            yticklabels[0] = '≤' + yticklabels[0]
        if extend in ['max','both']:
            yticklabels[-1] = '≥' + yticklabels[-1]
        if extend != 'neither':
            cax.set_yticks(yticks)
            cax.set_yticklabels(yticklabels)

    ### Return axes
    return cax, hax


def plotyearbymonth(dfs, plotcols=None, colors=None, 
    style='fill', lwforline=1, ls='-', figsize=(12,6), dpi=None,
    normalize=False, alpha=1, f=None, ax=None):
    """
    """
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    def monthifier(x):
        return pd.Timestamp('2001-01-{} {:02}:00'.format(x.day, x.hour))
    
    if (f is None) and (ax is None):
        f,ax=plt.subplots(12,1,figsize=figsize,sharex=True,sharey=True, dpi=dpi)
    else:
        pass

    for i, month in enumerate(months):
        if isinstance(dfs, pd.Series):
            plotcols = str(dfs.name)
            dfs = pd.DataFrame(dfs.rename(plotcols))
                
        if isinstance(dfs, pd.DataFrame):
            if plotcols is None:
                plotcols = dfs.columns.tolist()

            if isinstance(plotcols, str):
                dfplot = dfs.loc['{} {}'.format(month, dfs.index[0].year)][[plotcols]]
                if normalize:
                    dfplot = dfplot / dfplot.max()
                dfplot.index = dfplot.index.map(monthifier)

                if style in ['fill', 'fill_between', 'f']:
                    ax[i].fill_between(
                        dfplot.index, dfplot[plotcols].values, lw=0, alpha=alpha,
                        color=(colors if type(colors) in [str,mpl.colors.ListedColormap] 
                               else ('C0' if colors is None else colors[0])))
                elif style in ['line', 'l']:
                    ax[i].plot(
                        dfplot.index, dfplot[plotcols].values, lw=lwforline, alpha=alpha, ls=ls,
                        color=(colors if type(colors) in [str,mpl.colors.ListedColormap] 
                               else ('C0' if colors is None else colors[0])))
                    
            elif isinstance(plotcols, list):
                if isinstance(colors, str):
                    colors = [colors]*len(plotcols)
                elif colors is None:
                    colors = ['C{}'.format(i%10) for i in range(len(plotcols))]
                for j, plotcol in enumerate(plotcols):
                    dfplot = dfs.loc['{} {}'.format(month, dfs.index[0].year)][[plotcol]]
                    if normalize:
                        dfplot = dfplot / dfplot.max()
                    dfplot.index = dfplot.index.map(monthifier)

                    if style in ['fill', 'fill_between', 'f']:
                        ax[i].fill_between(dfplot.index, dfplot[plotcol].values, 
                                           lw=0, alpha=alpha, color=colors[j], label=plotcol)
                    elif style in ['line', 'l']:
                        ax[i].plot(dfplot.index, dfplot[plotcol].values, 
                                   lw=lwforline, alpha=alpha, ls=ls, color=colors[j], label=plotcol)
                                        
        ax[i].set_ylabel(month, rotation=0, ha='right', va='top')
        for which in ['left', 'right', 'top', 'bottom']:
                     ax[i].spines[which].set_visible(False)
        ax[i].tick_params(left=False,right=False,top=False,bottom=False)
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    ax[0].set_xlim(pd.to_datetime('2001-01-01 00:00'), pd.to_datetime('2001-02-01 00:00'))
    if normalize:
        ax[0].set_ylim(0, 1)
    else:
        pass
        # ax[0].set_ylim(0,dfs[plotcols].max())
    
    return f, ax


def plot_windrose(
    dfwind,
    dfdir,
    speedspacing=3,
    maxspeed=15,
    directionbinwidth=15,
    title=None,
    cmap=plt.cm.YlGnBu,
    ax=None,
    legend_kwargs={},
):
    ### Input parsing
    binedge = directionbinwidth / 2
    windbins = np.arange(0, maxspeed+0.001, speedspacing)
    directions = np.arange(0, 360, directionbinwidth)
    ### Get histograms
    dicthist = {}
    for direction in directions:
        if direction == 0:
            datetimes = dfdir.loc[
                ((direction - binedge + 360) % 360 <= dfdir)
                | (dfdir < (direction + binedge) % 360)
            ].index
        else:
            datetimes = dfdir.loc[
                ((direction - binedge + 360) % 360 <= dfdir)
                & (dfdir < (direction + binedge) % 360)
            ].index
        dicthist[direction] = np.histogram(dfwind.loc[datetimes], bins=windbins)[0]

    dfplot = (
        pd.DataFrame(dicthist, index=windbins[:-1])
        .rename_axis(index='windspeed', columns='direction')
        .cumsum()
    )

    colors = rainbowmapper(dfplot.index, cmap)

    ### Plot it
    if ax is None:
        plt.close()
        f,ax = plt.subplots(subplot_kw={'projection':'polar'})
    for windspeed, row in dfplot.iloc[::-1].iterrows():
        if windspeed == dfplot.index[-1]:
            label = f"{windspeed:.0f}+ m/s"
        else:
            label = f"{windspeed:.0f}–{windspeed+speedspacing:.0f} m/s"
        ax.bar(
            x=np.radians(row.index),
            height=row.values,
            width=np.radians(directionbinwidth),
            label=label,
            color=colors[windspeed],
            lw=0,
        )
    ## Legend
    if len(legend_kwargs) == 0:
        _legend_kwargs = {
            'loc':'upper left', 'bbox_to_anchor':(1, 1.1),
            'handletextpad':0.3, 'handlelength':0.7,
        }
    else:
        _legend_kwargs = legend_kwargs
    ax.legend(**_legend_kwargs)
    ## Formatting
    ax.set_yticklabels([])
    ax.set_xticks(np.radians(np.arange(0,360,45)))
    ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title)
    ax.grid(ls=':')

    return ax
