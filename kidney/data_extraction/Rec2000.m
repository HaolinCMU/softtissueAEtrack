%
% ABAQUS increment start record output to MATLAB
%
% Syntax
%     [#incrOut#,#StepHeading#]=Rec2000(#Rec#,zeros(1,0),zeros(1,0))
%
% Description
%     Read increment start record output from the results file generated
%     from the ABAQUS finite element software. The record key for increment
%     start record output is 2000. See section ''Results file output
%     format'' in ABAQUS Analysis User's manual for more details.
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
%     #incrOut# ([#n# x 11]) is an array containing the attributes of the
%         record key 2000 as follows:
%         Column  1�����Total time.
%         Column  2�����Step time.
%         Column  3�����Maximum creep strain-rate ratio (control of
%         solution-dependent amplitude) in Abaqus/Standard; currently not
%         used in Abaqus/Explicit.
%         Column  4�����Solution-dependent amplitude in Abaqus/Standard;
%         currently not used in Abaqus/Explicit.
%         Column  5�����Procedure type: gives a key to the step type.
%         Column  6�����Step number.
%         Column  7�����Increment number.
%         Column  8�����Linear perturbation flag in Abaqus/Standard: 0 if
%         general step, 1 if linear perturbation step; currently not used
%         in Abaqus/Explicit.
%         Column  9�����Load proportionality factor: nonzero only in static
%         Riks steps; currently not used in Abaqus/Explicit.
%         Column  10�����Frequency (cycles/time) in a steady-state dynamic
%         response analysis or steady-state transport angular velocity
%         (rad/time) in a steady-state transport analysis; currently not
%         used in Abaqus/Explicit.
%         Column  11�����Time increment.
%     #StepHeading# ([#n# x 80]) is a string array having 80 characters per
%         row. Each row contains the step subheading entered as the first
%         data line of the *STEP option (10A8�format). Equivalent to the
%         step description in Abaqus/CAE.
%     #n# is the number of increments.
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
