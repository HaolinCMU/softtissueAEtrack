%
% ABAQUS active degrees of freedom output to MATLAB
%
% Syntax
%     #out#=Rec1902(#Rec#,zeros(1,0),zeros(1,0))
%
% Description
%     Read active degrees of freedom output from the results file generated
%     from the ABAQUS finite element software. The asterisk (*) is replaced
%     by the name of the results file. The record key for active degrees of
%     freedom output is 1902. See section ''Results file output format'' in
%     ABAQUS Analysis User's manual for more details.
%     The following option with parameter has to be specified in the ABAQUS
%     input file for the results file to be created:
%         *FILE FORMAT, ASCII
%     Dummy input arguments are required for execution of this function, as
%     illustrated in the 'Input parameters' section below. These are used
%     for memory allocation issues and proper initialization of internal
%     variables.
%
% Input parameters
%     REQUIRED:
%     #Rec# (string) is a one-row string containing all the data inside the
%         Abaqus results file. The results file is generated by Abaqus
%         after the analysis has been completed.
%     DUMMY (in the following order, after REQUIRED):
%     zeros(1,0)
%     zeros(1,0)
%
% Output parameters
%     #out# ([1 x 30]) is an array containing the attributes of the record
%         key 1902 as follows:
%         Column  1  -  Location in nodal arrays of degree of freedom 1 (0
%         if DOF 1 is not active in the model).
%         Column  2  -  Location in nodal arrays of degree of freedom 2 (0
%         if DOF 2 is not active in the model).
%         Column  3  -  etc.
%
% _________________________________________________________________________
% Abaqus2Matlab - www.abaqus2matlab.com
% Copyright (c) 2017 by George Papazafeiropoulos
%
% If using this toolbox for research or industrial purposes, please cite:
% G. Papazafeiropoulos, M. Muniz-Calvente, E. Martinez-Paneda.
% Abaqus2Matlab: a suitable tool for finite element post-processing.
% Advances in Engineering Software. Vol 105. March 2017. Pages 9-16. (2017) 
% DOI:10.1016/j.advengsoft.2017.01.006

% Built-in function (Matlab R2012b)
