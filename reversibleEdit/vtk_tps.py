import vtk

from vtk.util.misc import vtkGetDataRoot
from vtk.util import numpy_support

import numpy as np


def create_vtk_thin_spline_warp(img,source_points, dst_points, use_inverse=True, show_grid=False):
    source_numpy_array = img
    VTK_data = numpy_support.numpy_to_vtk(num_array=source_numpy_array.ravel(), deep=True, array_type=numpy_support.get_vtk_array_type(img.dtype))
    VTK_data.SetNumberOfComponents(3)
    input_vtk_image = vtk.vtkImageData()
    input_vtk_image.SetDimensions(source_numpy_array.shape[1], source_numpy_array.shape[0],1)
    input_vtk_image.SetSpacing([1, 1, 1])
    input_vtk_image.SetOrigin([-1, -1, -1])
    input_vtk_image.GetPointData().SetScalars(VTK_data)
    
    # warp an image with a thin plate spline
    # first, create an image to warp
    imageGrid = vtk.vtkImageGridSource()
    imageGrid.SetGridSpacing(16,16,0)
    imageGrid.SetGridOrigin(0,0,0)
    imageGrid.SetDataExtent(0,img.shape[0],0,img.shape[1],0,0)
    imageGrid.SetDataScalarTypeToUnsignedChar()
    table = vtk.vtkLookupTable()
    table.SetTableRange(0,1)
    table.SetValueRange(1.0,0.0)
    table.SetSaturationRange(0.0,0.0)
    table.SetHueRange(0.0,0.0)
    table.SetAlphaRange(0.0,1.0)
    table.Build()
    alpha = vtk.vtkImageMapToColors()
    alpha.SetInputConnection(imageGrid.GetOutputPort())
    alpha.SetLookupTable(table)
    blend = vtk.vtkImageBlend()
    #blend.AddInputConnection(0,imageSource.GetOutputPort())
    blend.SetInputData(input_vtk_image)
    if show_grid:
        blend.AddInputConnection(0,alpha.GetOutputPort())
    # next, create a ThinPlateSpline transform
    p1 = vtk.vtkPoints()
    p2 = vtk.vtkPoints()
    p1.SetNumberOfPoints(source_points.shape[0])
    p2.SetNumberOfPoints(dst_points.shape[0])
    for i,p in enumerate(source_points):
        p1.SetPoint(i,p[0]*img.shape[0], p[1]*img.shape[1],0)
        dst_p = dst_points[i]
        p2.SetPoint(i,dst_p[0]*img.shape[0], dst_p[1]*img.shape[1],0)
        
    transform = vtk.vtkThinPlateSplineTransform()
    transform.SetSourceLandmarks(p1)
    transform.SetTargetLandmarks(p2)
    transform.SetBasisToR2LogR()
    # you must invert the transform before passing it to vtkImageReslice
    if use_inverse:
        transform.Inverse()
    reslice = vtk.vtkImageReslice()
    reslice.SetInputConnection(blend.GetOutputPort())
    reslice.SetResliceTransform(transform)
    reslice.SetInterpolationModeToLinear()
    reslice.Update()

    im = reslice.GetOutput()
    rows, cols, _ = im.GetDimensions()
    sc = im.GetPointData().GetScalars()
    numpy_data = numpy_support.vtk_to_numpy(sc)
    numpy_data = numpy_data.reshape(rows, cols, 3)
    numpy_data = numpy_data.transpose(0, 1, 2)

    return numpy_data



if __name__ == "__main__":
    source_points = np.array(
        ((0.45,0.5),(0.55,0.5),(0.5,0.45),(0.5,0.55))
    )

    destination_points = np.array(
        ((0.4,0.5),(0.6,0.5),(0.5,0.4),(0.5,0.6))
    )
    create_vtk_thin_spline_warp("../images/content/ferry.jpg", source_points, destination_points, "../output/warp_vtk_test.jpg")

