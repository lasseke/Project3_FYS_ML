'''
Add python files exclusive for Plotting here!
'''

def metricsBoxPlot(data_1, data_2, ticks, fig_size=(10,8), _save=False, savename="MetricsBoxPlot.png",\
                  title='Random Forest performance metrics - 10fold CV\n Custom implementation vs. scikit-learn',\
                  name_data_1='Custom', name_data_2='Sklearn'):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Function to define box plot colors
    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color='black',linewidth=1.5)
        plt.setp(bp)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color='red')

        for patch in bp['boxes']:
            patch.set_facecolor(color+"50")
        for whisker in bp['whiskers']: 
            whisker.set(linewidth=1.5)
        for cap in bp['caps']:
            cap.set(linewidth=1.5)
        for med in bp['medians']:
            med.set(linewidth=2)

    
    ### Plot results
    fig1, ax1 = plt.subplots(figsize=fig_size)
    # Colors
    col_data_1 = '#2ca25f'
    col_data_2 = '#2C7BB6'

    bpl = ax1.boxplot(data_1, positions=np.array(range(data_1.shape[1]))*2.0-0.4, sym='', widths=0.6, patch_artist=True)
    bpr = ax1.boxplot(data_2, positions=np.array(range(data_2.shape[1]))*2.0+0.4, sym='', widths=0.6, patch_artist=True)
    set_box_color(bpl, col_data_1) # colors are from http://colorbrewer2.org/
    set_box_color(bpr, col_data_2)


    #ax1.set_xlabel('Epoch', fontsize=18)
    #ax1.set_ylabel('Metric',fontsize=18)
    ax1.set_title(title, fontsize=22)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    
    ### LEGEND ###
    import matplotlib.patches as mpatches

    col_data_1 = mpatches.Patch(color=col_data_1+"50", label=name_data_1)
    col_data_2 = mpatches.Patch(color=col_data_2+"50", label=name_data_2)
    med_lin, = ax1.plot([], c='red', label='Median')
    ax1.legend(handles=[col_data_1, col_data_2, med_lin],fontsize=16,loc='best')
    #ax1.legend(fontsize=16,loc=4)

    ax1.set_xticks(range(0, len(ticks) * 2, 2))
    ax1.set_xticklabels(ticks, fontsize = 18)
    #ax1.set_ylim(0, 1.2)
    
    if _save:
        plt.savefig('./Results/FigureFiles/' + savename)
    
    plt.show()
    
    return fig1, ax1