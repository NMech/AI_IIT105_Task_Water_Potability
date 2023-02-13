# -*- coding: utf-8 -*-
from   BasicFigureTemplate import BasicFigure_Template
from   matplotlib          import colors
import seaborn           as sn
import numpy             as np
import matplotlib.pyplot as plt
#%%
class CorrelationMatrixPlot(BasicFigure_Template):
    
    def __init__(self,FigureProperties=["a3paper","pdf","landscape","white",0.5],FontSizes=[20.0,16.0,14.0,14.0]):
        """
        Initialization.
        """
        BasicFigure_Template.__init__(self,FigureProperties,FontSizes)
        
    def __mySymmetricColormap(self):
        """
        """
        color1 = plt.cm.coolwarm(np.linspace(1.0,0.0,128))
        color2 = plt.cm.coolwarm(np.linspace(0.0,1.0,128)) # This returns RGBA; convert:
        colour =  np.vstack((color1,color2))
        newColorMap = colors.LinearSegmentedColormap.from_list("my_colormap", colour)
        
        return newColorMap
        
    def PlotCorrelationHeatMaps(self,corrMatrix,colorMap="mySymmetric",Title="",
                                Rotations=[0.,45.],savePlot=["False","<filepath>","<filename>"],
                                showtriL=False,showtriU=False):
        """
        Function used for plotting the correlation heat maps.\n
        Keyword arguments:\n
            corrMatrix : The correlation matrix as produced by .corr method.\n
            colorMap   : Colormap to be used. Default value "mySymmetric". Other examples "coolwarm", "viridis" etx.\n
            Title      : The title to be used in the produced diagram.\n
            Rotations  : x,y-ticks rotations. Default values 0. and 45. degrees.\n
            savePlot   : list conatining the following.\n
                         * Save plot boolean.\n
                         * Filepath where the diagram will be saved.\n
                         * Filename (without the filetype) of the diagram to be plotted.\n
            showtriL   : Boolean. If True then only the lower triangular values of\n
                         the correlation matrix will be plotted.\n
            showtriU   : Boolean. If True then only the upper triangular values of\n
                         the correlation matrix will be plotted.\n             
        Return None.
        """
        size  = corrMatrix.shape[0]
        if showtriL == True and showtriU == False:
            maskP = np.tri(size,size,-1).T
        elif showtriU == True and showtriL == False:
            maskP = np.tri(size,size,-1)
        else:
            maskP = np.zeros((size,size))

        if colorMap == "mySymmetric":
            ColorMap = self.__mySymmetricColormap()
        else:
            ColorMap = colorMap 
        dim1,dim2   = self.FigureSize()   
        fig,ax      = plt.subplots(figsize=(dim1,dim2))
        
        sn.heatmap(corrMatrix,vmin=-1.,vmax=1., annot=True,linewidths=.5,linecolor="white",cmap=ColorMap,mask=maskP)
        ax.set_facecolor(f'xkcd:{self.FigureProperties[3]}')
        ax.patch.set_alpha(self.FigureProperties[4])
        ax.set_title(Title)
        ax.tick_params(axis="x",rotation=Rotations[0])
        ax.tick_params(axis="y",rotation=Rotations[1])
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig,savePlot[1],savePlot[2])
      
        return None