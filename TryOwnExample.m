## Copyright (C) 2020 Prajwal
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} testcase (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Prajwal <Prajwal@DESKTOP-IC712J7>
## Created: 2020-06-15

function TryOwnExample ( Theta1, Theta2)
  pkg load image;
  
  load('ex4data1.mat');
  
  %%load('ex4weights.mat');     ## using pretrained model for weights

  %%% ENTER NAME OF YOUR PHOTO HERE ! 
  mat = imread("9.jpg");
  
  mat = im2double(mat);
  
  mat=imresize(mat, [20 20]);
  
  size1 = size(mat);
  
  if (size(size1,2) == 3) 
    mat = rgb2gray(mat);
  endif
    
  imagesc(mat), colormap gray ;   ##Display image
  
  mat = mat(:)';

  fprintf(' Press enter to continue.\n');
  pause;  
  fprintf('\n\nIdentifying Number ... ');
      
  a1 = [1 mat];

  z2 = a1 * Theta1';

  a2 = sigmoid(z2);

  a2 = [ones(size(a2,1),1) a2];

  z3 = a2 * Theta2';

  a3 = sigmoid(z3);

  hofx = a3;

  prediction = hofx;
    
  [lol prediction] = max(hofx,[],2);
  
  fprintf('\n\nNumber Identified!  Press enter to continue.\n');
  pause;

  prediction  

endfunction
