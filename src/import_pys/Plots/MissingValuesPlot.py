# -*- coding: utf-8 -*-
from BasicFigureTemplate import BasicFigure_Template
import missingno         as msno 
import matplotlib.pyplot as plt
#%%
class MissingValuesPlot(BasicFigure_Template):
    
    def __init__(self,FigureProperties=["a3paper","pdf","landscape","white",0.5],
                 FontSizes=[20.0,16.0,14.0,10.0]):
        """
        Initialization.\n
        """
        BasicFigure_Template.__init__(self,FigureProperties,FontSizes)
        self.__color = (0.25, 0.25, 0.25)#(0.529, 0.808, 0.921)
        

    def MatrixPlot(self, data, savePlot=["False","<filepath>","<filename>"]):
        """
        Keyword arguments:\n
            data : [pd.DataFrame].\n
        """
        dim1,dim2   = self.FigureSize()   
        fig,ax      = plt.subplots(figsize=(dim1,dim2))

        msno.matrix(data, ax=ax, color=self.__color) 
        
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig, savePlot[1], savePlot[2])
          
        return fig, ax
    
    def BarPlot(self, data, savePlot=["False","<filepath>","<filename>"]):
        """
        Keyword arguments:\n
            data : [pd.DataFrame].\n
        """
        dim1,dim2   = self.FigureSize()   
        fig,ax      = plt.subplots(figsize=(dim1,dim2))
        
        msno.bar(data, ax=ax) 
        
        self.BackgroundColorOpacity(fig)
        
        if savePlot[0] == True:
            self.SaveFigure(fig, savePlot[1], savePlot[2])
          
        return fig, ax