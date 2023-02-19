# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import seaborn           as sn
from BasicFigureTemplate import BasicFigure_Template
#%%
FigureProperties= ["a3paper","pdf","landscape","white",0.5]
FontSizes       = [20.0,16.0,14.0,10.0]
#%%
def boxplot_potability(df, savePlot=["False","<filepath>","<filename>"]):
    """
    boxplot function. Tailored-made for Demokritos' coursework.
    """
    df = df.copy()
    df.loc[df["Potability"]==1, "Potability"] = "Potable"
    df.loc[df["Potability"]==0, "Potability"] = "non-Potable"
    
    basic_Obj_plot = BasicFigure_Template(FigureProperties,FontSizes)
    dim1,dim2      = basic_Obj_plot.FigureSize() 
    fig, ax = plt.subplots(3,3,figsize=(dim1,dim2))
    for i in range(3):
        for j in range(3):
            col = df.columns[i+3*j]
            sn.boxplot(data=df, x=col, y="Potability", ax = ax[i][j], 
                       notch=True, medianprops={"color": "red"})

            ax[i][j].set(ylabel=None)
            ax[i][j].set_yticklabels(ax[i][j].get_yticklabels(), rotation = 45.)
            ax[i][j].grid()
            
    basic_Obj_plot.BackgroundColorOpacity(fig)

    if savePlot[0] == True:
        basic_Obj_plot.SaveFigure(fig,savePlot[1],savePlot[2])
    
    return fig, ax