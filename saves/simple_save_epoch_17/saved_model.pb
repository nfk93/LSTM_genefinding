¬ã$
Ù8»8
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
 
ApplyAdagrad
var"T
accum"T
lr"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
update_slotsbool(

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype

ControlTrigger
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	

GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
:
Less
x"T
y"T
z
"
Ttype:
2	
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
;
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2

#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
A

StackPopV2

handle
elem"	elem_type"
	elem_typetype
X
StackPushV2

handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
S
StackV2
max_size

handle"
	elem_typetype"

stack_namestring 
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
Þ
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
Á
UnsortedSegmentSum	
data"T
segment_ids"Tindices
num_segments"Tnumsegments
output"T" 
Ttype:
2	"
Tindicestype:
2	" 
Tnumsegmentstype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.12.02
b'unknown'Ý¯"
d
PlaceholderPlaceholder*
dtype0*"
_output_shapes
:*
shape:
^
Placeholder_1Placeholder*
dtype0
*
_output_shapes

:*
shape
:
^
Placeholder_2Placeholder*
dtype0*
_output_shapes

:*
shape
:
w
prediction_xPlaceholder*
dtype0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
shape:ÿÿÿÿÿÿÿÿÿ
o
prediction_yPlaceholder*
dtype0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape:ÿÿÿÿÿÿÿÿÿ
e
random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
W
random_uniform/minConst*
valueB
 *JQZ¾*
dtype0*
_output_shapes
: 
W
random_uniform/maxConst*
valueB
 *JQZ>*
dtype0*
_output_shapes
: 

random_uniform/RandomUniformRandomUniformrandom_uniform/shape*
T0*
dtype0*
_output_shapes
:	*
seed2 *

seed 
b
random_uniform/subSubrandom_uniform/maxrandom_uniform/min*
T0*
_output_shapes
: 
u
random_uniform/mulMulrandom_uniform/RandomUniformrandom_uniform/sub*
T0*
_output_shapes
:	
g
random_uniformAddrandom_uniform/mulrandom_uniform/min*
T0*
_output_shapes
:	
w
w
VariableV2*
dtype0*
_output_shapes
:	*
shared_name *
	container *
shape:	

w/AssignAssignwrandom_uniform*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
U
w/readIdentityw*
T0*
_output_shapes
:	*
_class

loc:@w
`
random_uniform_1/shapeConst*
valueB:*
dtype0*
_output_shapes
:
Y
random_uniform_1/minConst*
valueB
 *×³]¿*
dtype0*
_output_shapes
: 
Y
random_uniform_1/maxConst*
valueB
 *×³]?*
dtype0*
_output_shapes
: 

random_uniform_1/RandomUniformRandomUniformrandom_uniform_1/shape*
T0*
dtype0*
_output_shapes
:*
seed2 *

seed 
h
random_uniform_1/subSubrandom_uniform_1/maxrandom_uniform_1/min*
T0*
_output_shapes
: 
v
random_uniform_1/mulMulrandom_uniform_1/RandomUniformrandom_uniform_1/sub*
T0*
_output_shapes
:
h
random_uniform_1Addrandom_uniform_1/mulrandom_uniform_1/min*
T0*
_output_shapes
:
m
b
VariableV2*
dtype0*
_output_shapes
:*
shared_name *
	container *
shape:

b/AssignAssignbrandom_uniform_1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
P
b/readIdentityb*
T0*
_output_shapes
:*
_class

loc:@b
]
DropoutWrapperInit/ConstConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_1Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
_
DropoutWrapperInit/Const_2Const*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 
`
Placeholder_3Placeholder*
dtype0*
_output_shapes
:	*
shape:	
`
Placeholder_4Placeholder*
dtype0*
_output_shapes
:	*
shape:	
`
Placeholder_5Placeholder*
dtype0*
_output_shapes
:	*
shape:	
`
Placeholder_6Placeholder*
dtype0*
_output_shapes
:	*
shape:	
`
Placeholder_7Placeholder*
dtype0*
_output_shapes
:	*
shape:	
`
Placeholder_8Placeholder*
dtype0*
_output_shapes
:	*
shape:	
J
rnn/RankConst*
value	B :*
dtype0*
_output_shapes
: 
Q
rnn/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
Q
rnn/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
f
	rnn/rangeRangernn/range/startrnn/Rankrnn/range/delta*
_output_shapes
:*

Tidx0
d
rnn/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
Q
rnn/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 


rnn/concatConcatV2rnn/concat/values_0	rnn/rangernn/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
m
rnn/transpose	TransposePlaceholder
rnn/concat*
T0*
Tperm0*"
_output_shapes
:
^
	rnn/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
a
rnn/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
rnn/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

rnn/strided_sliceStridedSlice	rnn/Shapernn/strided_slice/stackrnn/strided_slice/stack_1rnn/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
S
	rnn/ConstConst*
valueB:*
dtype0*
_output_shapes
:
V
rnn/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
S
rnn/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
}
rnn/concat_1ConcatV2	rnn/Constrnn/Const_1rnn/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
T
rnn/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
l
	rnn/zerosFillrnn/concat_1rnn/zeros/Const*
T0*
_output_shapes
:	*

index_type0
J
rnn/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ù
rnn/TensorArrayTensorArrayV3rnn/strided_slice*
dynamic_size( *
identical_element_shapes(*
element_shape:	*
_output_shapes

:: *
dtype0*/
tensor_array_namernn/dynamic_rnn/output_0*
clear_after_read(
ù
rnn/TensorArray_1TensorArrayV3rnn/strided_slice*
dynamic_size( *
identical_element_shapes(*
element_shape
:*
_output_shapes

:: *
dtype0*.
tensor_array_namernn/dynamic_rnn/input_0*
clear_after_read(
q
rnn/TensorArrayUnstack/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
t
*rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ì
$rnn/TensorArrayUnstack/strided_sliceStridedSlicernn/TensorArrayUnstack/Shape*rnn/TensorArrayUnstack/strided_slice/stack,rnn/TensorArrayUnstack/strided_slice/stack_1,rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
d
"rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
d
"rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
»
rnn/TensorArrayUnstack/rangeRange"rnn/TensorArrayUnstack/range/start$rnn/TensorArrayUnstack/strided_slice"rnn/TensorArrayUnstack/range/delta*
_output_shapes
:*

Tidx0
î
>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn/TensorArray_1rnn/TensorArrayUnstack/rangernn/transposernn/TensorArray_1:1*
T0*
_output_shapes
: * 
_class
loc:@rnn/transpose
O
rnn/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
Y
rnn/MaximumMaximumrnn/Maximum/xrnn/strided_slice*
T0*
_output_shapes
: 
W
rnn/MinimumMinimumrnn/strided_slicernn/Maximum*
T0*
_output_shapes
: 
]
rnn/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
­
rnn/while/EnterEnterrnn/while/iteration_counter*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
: 

rnn/while/Enter_1Enterrnn/time*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
: 
¥
rnn/while/Enter_2Enterrnn/TensorArray:1*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
: 
ª
rnn/while/Enter_3EnterPlaceholder_3*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
ª
rnn/while/Enter_4EnterPlaceholder_6*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
ª
rnn/while/Enter_5EnterPlaceholder_4*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
ª
rnn/while/Enter_6EnterPlaceholder_7*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
ª
rnn/while/Enter_7EnterPlaceholder_5*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
ª
rnn/while/Enter_8EnterPlaceholder_8*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
:	
n
rnn/while/MergeMergernn/while/Enterrnn/while/NextIteration*
T0*
N*
_output_shapes
: : 
t
rnn/while/Merge_1Mergernn/while/Enter_1rnn/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
t
rnn/while/Merge_2Mergernn/while/Enter_2rnn/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
}
rnn/while/Merge_3Mergernn/while/Enter_3rnn/while/NextIteration_3*
T0*
N*!
_output_shapes
:	: 
}
rnn/while/Merge_4Mergernn/while/Enter_4rnn/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 
}
rnn/while/Merge_5Mergernn/while/Enter_5rnn/while/NextIteration_5*
T0*
N*!
_output_shapes
:	: 
}
rnn/while/Merge_6Mergernn/while/Enter_6rnn/while/NextIteration_6*
T0*
N*!
_output_shapes
:	: 
}
rnn/while/Merge_7Mergernn/while/Enter_7rnn/while/NextIteration_7*
T0*
N*!
_output_shapes
:	: 
}
rnn/while/Merge_8Mergernn/while/Enter_8rnn/while/NextIteration_8*
T0*
N*!
_output_shapes
:	: 
^
rnn/while/LessLessrnn/while/Mergernn/while/Less/Enter*
T0*
_output_shapes
: 
¨
rnn/while/Less/EnterEnterrnn/strided_slice*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
: 
d
rnn/while/Less_1Lessrnn/while/Merge_1rnn/while/Less_1/Enter*
T0*
_output_shapes
: 
¤
rnn/while/Less_1/EnterEnterrnn/Minimum*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
: 
\
rnn/while/LogicalAnd
LogicalAndrnn/while/Lessrnn/while/Less_1*
_output_shapes
: 
L
rnn/while/LoopCondLoopCondrnn/while/LogicalAnd*
_output_shapes
: 

rnn/while/SwitchSwitchrnn/while/Mergernn/while/LoopCond*
T0*
_output_shapes
: : *"
_class
loc:@rnn/while/Merge

rnn/while/Switch_1Switchrnn/while/Merge_1rnn/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_1

rnn/while/Switch_2Switchrnn/while/Merge_2rnn/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@rnn/while/Merge_2

rnn/while/Switch_3Switchrnn/while/Merge_3rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_3

rnn/while/Switch_4Switchrnn/while/Merge_4rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_4

rnn/while/Switch_5Switchrnn/while/Merge_5rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_5

rnn/while/Switch_6Switchrnn/while/Merge_6rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_6

rnn/while/Switch_7Switchrnn/while/Merge_7rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_7

rnn/while/Switch_8Switchrnn/while/Merge_8rnn/while/LoopCond*
T0**
_output_shapes
:	:	*$
_class
loc:@rnn/while/Merge_8
S
rnn/while/IdentityIdentityrnn/while/Switch:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_1Identityrnn/while/Switch_1:1*
T0*
_output_shapes
: 
W
rnn/while/Identity_2Identityrnn/while/Switch_2:1*
T0*
_output_shapes
: 
`
rnn/while/Identity_3Identityrnn/while/Switch_3:1*
T0*
_output_shapes
:	
`
rnn/while/Identity_4Identityrnn/while/Switch_4:1*
T0*
_output_shapes
:	
`
rnn/while/Identity_5Identityrnn/while/Switch_5:1*
T0*
_output_shapes
:	
`
rnn/while/Identity_6Identityrnn/while/Switch_6:1*
T0*
_output_shapes
:	
`
rnn/while/Identity_7Identityrnn/while/Switch_7:1*
T0*
_output_shapes
:	
`
rnn/while/Identity_8Identityrnn/while/Switch_8:1*
T0*
_output_shapes
:	
f
rnn/while/add/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
rnn/while/addAddrnn/while/Identityrnn/while/add/y*
T0*
_output_shapes
: 
»
rnn/while/TensorArrayReadV3TensorArrayReadV3!rnn/while/TensorArrayReadV3/Enterrnn/while/Identity_1#rnn/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes

:
¹
!rnn/while/TensorArrayReadV3/EnterEnterrnn/TensorArray_1*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
ä
#rnn/while/TensorArrayReadV3/Enter_1Enter>rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
: 
Û
Krnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *ÓÅ½*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *ÓÅ=*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Á
Srnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Æ
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/subSubIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/maxIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ú
Irnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulMulSrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Ì
Ernn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniformAddIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/mulIrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
á
*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Á
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AssignAssign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelErnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel

/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/readIdentity*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel*
T0* 
_output_shapes
:

Æ
:rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
Ó
(rnn/multi_rnn_cell/cell_0/lstm_cell/bias
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
«
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AssignAssign(rnn/multi_rnn_cell/cell_0/lstm_cell/bias:rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias

-rnn/multi_rnn_cell/cell_0/lstm_cell/bias/readIdentity(rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
T0*
_output_shapes	
:

9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
í
4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concatConcatV2rnn/while/TensorArrayReadV3rnn/while/Identity_49rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	*

Tidx0

4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMulMatMul4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat:rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ö
:rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context* 
_output_shapes
:

ô
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAddBiasAdd4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul;rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ð
;rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/EnterEnter-rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes	
:

3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/splitSplit=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Î
1rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/addAdd5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:23rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y*
T0*
_output_shapes
:	

5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/SigmoidSigmoid1rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add*
T0*
_output_shapes
:	
¯
1rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mulMul5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoidrnn/while/Identity_3*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split*
T0*
_output_shapes
:	

2rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/TanhTanh5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1*
T0*
_output_shapes
:	
Ñ
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_12rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh*
T0*
_output_shapes
:	
Ì
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1Add1rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1Tanh3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2Mul7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_24rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1*
T0*
_output_shapes
:	
Û
Krnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *óµ½*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *óµ=*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Á
Srnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Æ
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/subSubIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/maxIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ú
Irnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulMulSrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Ì
Ernn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniformAddIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/mulIrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
á
*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Á
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AssignAssign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelErnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel

/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/readIdentity*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel*
T0* 
_output_shapes
:

Æ
:rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
Ó
(rnn/multi_rnn_cell/cell_1/lstm_cell/bias
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
«
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AssignAssign(rnn/multi_rnn_cell/cell_1/lstm_cell/bias:rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias

-rnn/multi_rnn_cell/cell_1/lstm_cell/bias/readIdentity(rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
T0*
_output_shapes	
:

9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concatConcatV23rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2rnn/while/Identity_69rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	*

Tidx0

4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMulMatMul4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat:rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ö
:rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context* 
_output_shapes
:

ô
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAddBiasAdd4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul;rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ð
;rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/EnterEnter-rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes	
:

3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/splitSplit=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Î
1rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/addAdd5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:23rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y*
T0*
_output_shapes
:	

5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/SigmoidSigmoid1rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add*
T0*
_output_shapes
:	
¯
1rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mulMul5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoidrnn/while/Identity_5*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split*
T0*
_output_shapes
:	

2rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/TanhTanh5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1*
T0*
_output_shapes
:	
Ñ
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_12rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh*
T0*
_output_shapes
:	
Ì
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1Add1rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1Tanh3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2Mul7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_24rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1*
T0*
_output_shapes
:	
Û
Krnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/minConst*
valueB
 *óµ½*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Í
Irnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/maxConst*
valueB
 *óµ=*
dtype0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Á
Srnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformKrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/shape*
seed2 *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel*
T0* 
_output_shapes
:
*
dtype0*

seed 
Æ
Irnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/subSubIrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/maxIrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
: *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Ú
Irnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/mulMulSrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/RandomUniformIrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Ì
Ernn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniformAddIrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/mulIrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform/min*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
á
*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Á
1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AssignAssign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelErnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/readIdentity*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel*
T0* 
_output_shapes
:

Æ
:rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Initializer/zerosConst*
valueB*    *
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
Ó
(rnn/multi_rnn_cell/cell_2/lstm_cell/bias
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
«
/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AssignAssign(rnn/multi_rnn_cell/cell_2/lstm_cell/bias:rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Initializer/zeros*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias

-rnn/multi_rnn_cell/cell_2/lstm_cell/bias/readIdentity(rnn/multi_rnn_cell/cell_2/lstm_cell/bias*
T0*
_output_shapes	
:

9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat/axisConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concatConcatV23rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2rnn/while/Identity_89rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat/axis*
T0*
N*
_output_shapes
:	*

Tidx0

4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMulMatMul4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat:rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ö
:rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context* 
_output_shapes
:

ô
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAddBiasAdd4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul;rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ð
;rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/EnterEnter-rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes	
:

3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/ConstConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split/split_dimConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/splitSplit=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split/split_dim5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add/yConst^rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Î
1rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/addAdd5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:23rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add/y*
T0*
_output_shapes
:	

5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/SigmoidSigmoid1rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add*
T0*
_output_shapes
:	
¯
1rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mulMul5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoidrnn/while/Identity_7*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split*
T0*
_output_shapes
:	

2rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/TanhTanh5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:1*
T0*
_output_shapes
:	
Ñ
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1Mul7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_12rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh*
T0*
_output_shapes
:	
Ì
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1Add1rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1Tanh3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2Mul7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_24rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1*
T0*
_output_shapes
:	
u
rnn/while/dropout/keep_probConst^rnn/while/Identity*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 
}
rnn/while/dropout/ShapeConst^rnn/while/Identity*
valueB"      *
dtype0*
_output_shapes
:
~
$rnn/while/dropout/random_uniform/minConst^rnn/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 
~
$rnn/while/dropout/random_uniform/maxConst^rnn/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¨
.rnn/while/dropout/random_uniform/RandomUniformRandomUniformrnn/while/dropout/Shape*
T0*
dtype0*
_output_shapes
:	*
seed2 *

seed 

$rnn/while/dropout/random_uniform/subSub$rnn/while/dropout/random_uniform/max$rnn/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
«
$rnn/while/dropout/random_uniform/mulMul.rnn/while/dropout/random_uniform/RandomUniform$rnn/while/dropout/random_uniform/sub*
T0*
_output_shapes
:	

 rnn/while/dropout/random_uniformAdd$rnn/while/dropout/random_uniform/mul$rnn/while/dropout/random_uniform/min*
T0*
_output_shapes
:	

rnn/while/dropout/addAddrnn/while/dropout/keep_prob rnn/while/dropout/random_uniform*
T0*
_output_shapes
:	
a
rnn/while/dropout/FloorFloorrnn/while/dropout/add*
T0*
_output_shapes
:	

rnn/while/dropout/divRealDiv3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2rnn/while/dropout/keep_prob*
T0*
_output_shapes
:	
v
rnn/while/dropout/mulMulrnn/while/dropout/divrnn/while/dropout/Floor*
T0*
_output_shapes
:	

-rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn/while/Identity_1rnn/while/dropout/mulrnn/while/Identity_2*
T0*
_output_shapes
: *(
_class
loc:@rnn/while/dropout/mul
ó
3rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn/TensorArray*
is_constant(*(
_class
loc:@rnn/while/dropout/mul*
T0*
parallel_iterations *'

frame_namernn/while/while_context*
_output_shapes
:
h
rnn/while/add_1/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
rnn/while/add_1Addrnn/while/Identity_1rnn/while/add_1/y*
T0*
_output_shapes
: 
X
rnn/while/NextIterationNextIterationrnn/while/add*
T0*
_output_shapes
: 
\
rnn/while/NextIteration_1NextIterationrnn/while/add_1*
T0*
_output_shapes
: 
z
rnn/while/NextIteration_2NextIteration-rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

rnn/while/NextIteration_3NextIteration3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1*
T0*
_output_shapes
:	

rnn/while/NextIteration_4NextIteration3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2*
T0*
_output_shapes
:	

rnn/while/NextIteration_5NextIteration3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1*
T0*
_output_shapes
:	

rnn/while/NextIteration_6NextIteration3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2*
T0*
_output_shapes
:	

rnn/while/NextIteration_7NextIteration3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1*
T0*
_output_shapes
:	

rnn/while/NextIteration_8NextIteration3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2*
T0*
_output_shapes
:	
I
rnn/while/ExitExitrnn/while/Switch*
T0*
_output_shapes
: 
M
rnn/while/Exit_1Exitrnn/while/Switch_1*
T0*
_output_shapes
: 
M
rnn/while/Exit_2Exitrnn/while/Switch_2*
T0*
_output_shapes
: 
V
rnn/while/Exit_3Exitrnn/while/Switch_3*
T0*
_output_shapes
:	
V
rnn/while/Exit_4Exitrnn/while/Switch_4*
T0*
_output_shapes
:	
V
rnn/while/Exit_5Exitrnn/while/Switch_5*
T0*
_output_shapes
:	
V
rnn/while/Exit_6Exitrnn/while/Switch_6*
T0*
_output_shapes
:	
V
rnn/while/Exit_7Exitrnn/while/Switch_7*
T0*
_output_shapes
:	
V
rnn/while/Exit_8Exitrnn/while/Switch_8*
T0*
_output_shapes
:	

&rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn/TensorArrayrnn/while/Exit_2*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

 rnn/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *"
_class
loc:@rnn/TensorArray
ä
rnn/TensorArrayStack/rangeRange rnn/TensorArrayStack/range/start&rnn/TensorArrayStack/TensorArraySizeV3 rnn/TensorArrayStack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0*"
_class
loc:@rnn/TensorArray
ò
(rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn/TensorArrayrnn/TensorArrayStack/rangernn/while/Exit_2*
element_shape:	*
dtype0*#
_output_shapes
:*"
_class
loc:@rnn/TensorArray
\
rnn/Const_2Const*
valueB"      *
dtype0*
_output_shapes
:
V
rnn/Const_3Const*
valueB:*
dtype0*
_output_shapes
:
L

rnn/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
S
rnn/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
S
rnn/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
n
rnn/range_1Rangernn/range_1/start
rnn/Rank_1rnn/range_1/delta*
_output_shapes
:*

Tidx0
f
rnn/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
S
rnn/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

rnn/concat_2ConcatV2rnn/concat_2/values_0rnn/range_1rnn/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0

rnn/transpose_1	Transpose(rnn/TensorArrayStack/TensorArrayGatherV3rnn/concat_2*
T0*
Tperm0*#
_output_shapes
:
j
einsum/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

einsum/transpose	Transposernn/transpose_1einsum/transpose/perm*
T0*
Tperm0*#
_output_shapes
:
h
einsum/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
w
einsum/transpose_1	Transposew/readeinsum/transpose_1/perm*
T0*
Tperm0*
_output_shapes
:	
e
einsum/Reshape/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
z
einsum/ReshapeReshapeeinsum/transposeeinsum/Reshape/shape*
T0*
Tshape0* 
_output_shapes
:


einsum/MatMulMatMuleinsum/Reshapeeinsum/transpose_1*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
k
einsum/Reshape_1/shapeConst*!
valueB"         *
dtype0*
_output_shapes
:
}
einsum/Reshape_1Reshapeeinsum/MatMuleinsum/Reshape_1/shape*
T0*
Tshape0*"
_output_shapes
:
l
einsum/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:

einsum/transpose_2	Transposeeinsum/Reshape_1einsum/transpose_2/perm*
T0*
Tperm0*"
_output_shapes
:
S
addAddeinsum/transpose_2b/read*
T0*"
_output_shapes
:
D
SoftmaxSoftmaxadd*
T0*"
_output_shapes
:
R
ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
s
ArgMaxArgMaxSoftmaxArgMax/dimension*
output_type0	*
T0*
_output_shapes

:*

Tidx0
\
CastCastArgMax*

DstT0*
_output_shapes

:*
Truncate( *

SrcT0	
L
EqualEqualCastPlaceholder_2*
T0*
_output_shapes

:
z
)SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:

+SparseSoftmaxCrossEntropyWithLogits/Shape_1Const*!
valueB"         *
dtype0*
_output_shapes
:
j
(SparseSoftmaxCrossEntropyWithLogits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
k
)SparseSoftmaxCrossEntropyWithLogits/sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
¤
'SparseSoftmaxCrossEntropyWithLogits/subSub(SparseSoftmaxCrossEntropyWithLogits/Rank)SparseSoftmaxCrossEntropyWithLogits/sub/y*
T0*
_output_shapes
: 
k
)SparseSoftmaxCrossEntropyWithLogits/add/yConst*
value	B :*
dtype0*
_output_shapes
: 
£
'SparseSoftmaxCrossEntropyWithLogits/addAdd'SparseSoftmaxCrossEntropyWithLogits/sub)SparseSoftmaxCrossEntropyWithLogits/add/y*
T0*
_output_shapes
: 
¢
7SparseSoftmaxCrossEntropyWithLogits/strided_slice/stackPack'SparseSoftmaxCrossEntropyWithLogits/sub*

axis *
T0*
N*
_output_shapes
:
¤
9SparseSoftmaxCrossEntropyWithLogits/strided_slice/stack_1Pack'SparseSoftmaxCrossEntropyWithLogits/add*

axis *
T0*
N*
_output_shapes
:

9SparseSoftmaxCrossEntropyWithLogits/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¯
1SparseSoftmaxCrossEntropyWithLogits/strided_sliceStridedSlice+SparseSoftmaxCrossEntropyWithLogits/Shape_17SparseSoftmaxCrossEntropyWithLogits/strided_slice/stack9SparseSoftmaxCrossEntropyWithLogits/strided_slice/stack_19SparseSoftmaxCrossEntropyWithLogits/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
~
3SparseSoftmaxCrossEntropyWithLogits/Reshape/shape/0Const*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
Û
1SparseSoftmaxCrossEntropyWithLogits/Reshape/shapePack3SparseSoftmaxCrossEntropyWithLogits/Reshape/shape/01SparseSoftmaxCrossEntropyWithLogits/strided_slice*

axis *
T0*
N*
_output_shapes
:
¦
+SparseSoftmaxCrossEntropyWithLogits/ReshapeReshapeadd1SparseSoftmaxCrossEntropyWithLogits/Reshape/shape*
T0*
Tshape0*
_output_shapes
:	

3SparseSoftmaxCrossEntropyWithLogits/Reshape_1/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:
°
-SparseSoftmaxCrossEntropyWithLogits/Reshape_1ReshapePlaceholder_23SparseSoftmaxCrossEntropyWithLogits/Reshape_1/shape*
T0*
Tshape0*
_output_shapes	
:

GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogits+SparseSoftmaxCrossEntropyWithLogits/Reshape-SparseSoftmaxCrossEntropyWithLogits/Reshape_1*
T0*&
_output_shapes
::	*
Tlabels0
ã
-SparseSoftmaxCrossEntropyWithLogits/Reshape_2ReshapeGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits)SparseSoftmaxCrossEntropyWithLogits/Shape*
T0*
Tshape0*
_output_shapes

:
c
boolean_mask/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
j
 boolean_mask/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
l
"boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
l
"boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¾
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape boolean_mask/strided_slice/stack"boolean_mask/strided_slice/stack_1"boolean_mask/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask *
new_axis_mask *

begin_mask *
_output_shapes
:*
ellipsis_mask 
m
#boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0*
_output_shapes
:

boolean_mask/ProdProdboolean_mask/strided_slice#boolean_mask/Prod/reduction_indices*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
e
boolean_mask/Shape_1Const*
valueB"      *
dtype0*
_output_shapes
:
l
"boolean_mask/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
n
$boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
n
$boolean_mask/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Æ
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1"boolean_mask/strided_slice_1/stack$boolean_mask/strided_slice_1/stack_1$boolean_mask/strided_slice_1/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask *
new_axis_mask *

begin_mask*
_output_shapes
: *
ellipsis_mask 
e
boolean_mask/Shape_2Const*
valueB"      *
dtype0*
_output_shapes
:
l
"boolean_mask/strided_slice_2/stackConst*
valueB:*
dtype0*
_output_shapes
:
n
$boolean_mask/strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
n
$boolean_mask/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Æ
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2"boolean_mask/strided_slice_2/stack$boolean_mask/strided_slice_2/stack_1$boolean_mask/strided_slice_2/stack_2*
Index0*
end_mask*
T0*
shrink_axis_mask *
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
q
boolean_mask/concat/values_1Packboolean_mask/Prod*

axis *
T0*
N*
_output_shapes
:
Z
boolean_mask/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
Í
boolean_mask/concatConcatV2boolean_mask/strided_slice_1boolean_mask/concat/values_1boolean_mask/strided_slice_2boolean_mask/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

boolean_mask/ReshapeReshape-SparseSoftmaxCrossEntropyWithLogits/Reshape_2boolean_mask/concat*
T0*
Tshape0*
_output_shapes	
:
o
boolean_mask/Reshape_1/shapeConst*
valueB:
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
:

boolean_mask/Reshape_1ReshapePlaceholder_1boolean_mask/Reshape_1/shape*
T0
*
Tshape0*
_output_shapes	
:
e
boolean_mask/WhereWhereboolean_mask/Reshape_1*
T0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
x
boolean_mask/SqueezeSqueezeboolean_mask/Where*
T0	*
squeeze_dims
*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\
boolean_mask/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
º
boolean_mask/GatherV2GatherV2boolean_mask/Reshapeboolean_mask/Squeezeboolean_mask/GatherV2/axis*
Taxis0*
Tparams0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Tindices0	
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
h
MeanMeanboolean_mask/GatherV2Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
S
gradients/f_countConst*
value	B : *
dtype0*
_output_shapes
: 
§
gradients/f_count_1Entergradients/f_count*
T0*
parallel_iterations *
is_constant( *'

frame_namernn/while/while_context*
_output_shapes
: 
r
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N*
_output_shapes
: : 
b
gradients/SwitchSwitchgradients/Mergernn/while/LoopCond*
T0*
_output_shapes
: : 
f
gradients/Add/yConst^rnn/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0*
_output_shapes
: 
í
gradients/NextIterationNextIterationgradients/Add[^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV25^gradients/rnn/while/dropout/div_grad/Neg/StackPushV25^gradients/rnn/while/dropout/mul_grad/Mul/StackPushV27^gradients/rnn/while/dropout/mul_grad/Mul_1/StackPushV2Y^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2Y^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2Y^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPushV2U^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPushV2Q^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPushV2S^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0*
_output_shapes
: 
N
gradients/f_count_2Exitgradients/Switch*
T0*
_output_shapes
: 
S
gradients/b_countConst*
value	B :*
dtype0*
_output_shapes
: 
³
gradients/b_count_1Entergradients/f_count_2*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
v
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N*
_output_shapes
: : 
x
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
º
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
O
gradients/b_count_2LoopCondgradients/GreaterEqual*
_output_shapes
: 
g
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0*
_output_shapes
: : 
i
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0*
_output_shapes
: 
²
gradients/NextIteration_1NextIterationgradients/SubV^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0*
_output_shapes
: 
P
gradients/b_count_3Exitgradients/Switch_1*
T0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
n
gradients/Mean_grad/ShapeShapeboolean_mask/GatherV2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tmultiples0
p
gradients/Mean_grad/Shape_1Shapeboolean_mask/GatherV2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

DstT0*
_output_shapes
: *
Truncate( *

SrcT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

*gradients/boolean_mask/GatherV2_grad/ShapeConst*
valueB	R*
dtype0	*
_output_shapes
:*'
_class
loc:@boolean_mask/Reshape
Í
,gradients/boolean_mask/GatherV2_grad/ToInt32Cast*gradients/boolean_mask/GatherV2_grad/Shape*

DstT0*
_output_shapes
:*
Truncate( *

SrcT0	*'
_class
loc:@boolean_mask/Reshape
x
)gradients/boolean_mask/GatherV2_grad/SizeSizeboolean_mask/Squeeze*
T0	*
out_type0*
_output_shapes
: 
u
3gradients/boolean_mask/GatherV2_grad/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
Î
/gradients/boolean_mask/GatherV2_grad/ExpandDims
ExpandDims)gradients/boolean_mask/GatherV2_grad/Size3gradients/boolean_mask/GatherV2_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:

8gradients/boolean_mask/GatherV2_grad/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:

:gradients/boolean_mask/GatherV2_grad/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:

:gradients/boolean_mask/GatherV2_grad/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¶
2gradients/boolean_mask/GatherV2_grad/strided_sliceStridedSlice,gradients/boolean_mask/GatherV2_grad/ToInt328gradients/boolean_mask/GatherV2_grad/strided_slice/stack:gradients/boolean_mask/GatherV2_grad/strided_slice/stack_1:gradients/boolean_mask/GatherV2_grad/strided_slice/stack_2*
Index0*
end_mask*
T0*
shrink_axis_mask *
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
r
0gradients/boolean_mask/GatherV2_grad/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

+gradients/boolean_mask/GatherV2_grad/concatConcatV2/gradients/boolean_mask/GatherV2_grad/ExpandDims2gradients/boolean_mask/GatherV2_grad/strided_slice0gradients/boolean_mask/GatherV2_grad/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
½
,gradients/boolean_mask/GatherV2_grad/ReshapeReshapegradients/Mean_grad/truediv+gradients/boolean_mask/GatherV2_grad/concat*
T0*
Tshape0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¼
.gradients/boolean_mask/GatherV2_grad/Reshape_1Reshapeboolean_mask/Squeeze/gradients/boolean_mask/GatherV2_grad/ExpandDims*
T0	*
Tshape0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
z
)gradients/boolean_mask/Reshape_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:

?gradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:

Agradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:

Agradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
Ð
9gradients/boolean_mask/Reshape_grad/Reshape/strided_sliceStridedSlice,gradients/boolean_mask/GatherV2_grad/ToInt32?gradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stackAgradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stack_1Agradients/boolean_mask/Reshape_grad/Reshape/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
«
2gradients/boolean_mask/Reshape_grad/Reshape/tensorUnsortedSegmentSum,gradients/boolean_mask/GatherV2_grad/Reshape.gradients/boolean_mask/GatherV2_grad/Reshape_19gradients/boolean_mask/Reshape_grad/Reshape/strided_slice*
T0*
_output_shapes	
:*
Tindices0	*
Tnumsegments0
Ì
+gradients/boolean_mask/Reshape_grad/ReshapeReshape2gradients/boolean_mask/Reshape_grad/Reshape/tensor)gradients/boolean_mask/Reshape_grad/Shape*
T0*
Tshape0*
_output_shapes

:

Bgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_2_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
ô
Dgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_2_grad/ReshapeReshape+gradients/boolean_mask/Reshape_grad/ReshapeBgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_2_grad/Shape*
T0*
Tshape0*
_output_shapes	
:

gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes
:	
¥
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*´
message¨¥Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()*
_output_shapes
:	
°
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: 
Ò
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsDgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_2_grad/Reshapeegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*

Tdim0*
_output_shapes
:	
Ö
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*
_output_shapes
:	

@gradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
¦
Bgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_grad/ReshapeReshapeZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul@gradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
m
gradients/add_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
d
gradients/add_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
´
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
Õ
gradients/add_grad/SumSumBgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*"
_output_shapes
:*

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*"
_output_shapes
:
Ñ
gradients/add_grad/Sum_1SumBgradients/SparseSoftmaxCrossEntropyWithLogits/Reshape_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Õ
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*
T0*"
_output_shapes
:*-
_class#
!loc:@gradients/add_grad/Reshape
Ó
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*
_output_shapes
:*/
_class%
#!loc:@gradients/add_grad/Reshape_1

3gradients/einsum/transpose_2_grad/InvertPermutationInvertPermutationeinsum/transpose_2/perm*
T0*
_output_shapes
:
Ô
+gradients/einsum/transpose_2_grad/transpose	Transpose+gradients/add_grad/tuple/control_dependency3gradients/einsum/transpose_2_grad/InvertPermutation*
T0*
Tperm0*"
_output_shapes
:
v
%gradients/einsum/Reshape_1_grad/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
¾
'gradients/einsum/Reshape_1_grad/ReshapeReshape+gradients/einsum/transpose_2_grad/transpose%gradients/einsum/Reshape_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
»
#gradients/einsum/MatMul_grad/MatMulMatMul'gradients/einsum/Reshape_1_grad/Reshapeeinsum/transpose_1*
T0*
transpose_b(*
transpose_a( * 
_output_shapes
:

¸
%gradients/einsum/MatMul_grad/MatMul_1MatMuleinsum/Reshape'gradients/einsum/Reshape_1_grad/Reshape*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	

-gradients/einsum/MatMul_grad/tuple/group_depsNoOp$^gradients/einsum/MatMul_grad/MatMul&^gradients/einsum/MatMul_grad/MatMul_1
ù
5gradients/einsum/MatMul_grad/tuple/control_dependencyIdentity#gradients/einsum/MatMul_grad/MatMul.^gradients/einsum/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*6
_class,
*(loc:@gradients/einsum/MatMul_grad/MatMul
þ
7gradients/einsum/MatMul_grad/tuple/control_dependency_1Identity%gradients/einsum/MatMul_grad/MatMul_1.^gradients/einsum/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*8
_class.
,*loc:@gradients/einsum/MatMul_grad/MatMul_1
x
#gradients/einsum/Reshape_grad/ShapeConst*!
valueB"         *
dtype0*
_output_shapes
:
È
%gradients/einsum/Reshape_grad/ReshapeReshape5gradients/einsum/MatMul_grad/tuple/control_dependency#gradients/einsum/Reshape_grad/Shape*
T0*
Tshape0*#
_output_shapes
:

3gradients/einsum/transpose_1_grad/InvertPermutationInvertPermutationeinsum/transpose_1/perm*
T0*
_output_shapes
:
Ý
+gradients/einsum/transpose_1_grad/transpose	Transpose7gradients/einsum/MatMul_grad/tuple/control_dependency_13gradients/einsum/transpose_1_grad/InvertPermutation*
T0*
Tperm0*
_output_shapes
:	

1gradients/einsum/transpose_grad/InvertPermutationInvertPermutationeinsum/transpose/perm*
T0*
_output_shapes
:
Ë
)gradients/einsum/transpose_grad/transpose	Transpose%gradients/einsum/Reshape_grad/Reshape1gradients/einsum/transpose_grad/InvertPermutation*
T0*
Tperm0*#
_output_shapes
:
x
0gradients/rnn/transpose_1_grad/InvertPermutationInvertPermutationrnn/concat_2*
T0*
_output_shapes
:
Í
(gradients/rnn/transpose_1_grad/transpose	Transpose)gradients/einsum/transpose_grad/transpose0gradients/rnn/transpose_1_grad/InvertPermutation*
T0*
Tperm0*#
_output_shapes
:
ê
Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3rnn/TensorArrayrnn/while/Exit_2*
_output_shapes

:: *
source	gradients*"
_class
loc:@rnn/TensorArray

Ugradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityrnn/while/Exit_2Z^gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *"
_class
loc:@rnn/TensorArray

_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3Ygradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3rnn/TensorArrayStack/range(gradients/rnn/transpose_1_grad/transposeUgradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0*
_output_shapes
: 
f
gradients/zerosConst*
valueB	*    *
dtype0*
_output_shapes
:	
h
gradients/zeros_1Const*
valueB	*    *
dtype0*
_output_shapes
:	
h
gradients/zeros_2Const*
valueB	*    *
dtype0*
_output_shapes
:	
h
gradients/zeros_3Const*
valueB	*    *
dtype0*
_output_shapes
:	
h
gradients/zeros_4Const*
valueB	*    *
dtype0*
_output_shapes
:	
h
gradients/zeros_5Const*
valueB	*    *
dtype0*
_output_shapes
:	

&gradients/rnn/while/Exit_2_grad/b_exitEnter_gradients/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
: 
Ë
&gradients/rnn/while/Exit_3_grad/b_exitEntergradients/zeros*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
Í
&gradients/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_1*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
Í
&gradients/rnn/while/Exit_5_grad/b_exitEntergradients/zeros_2*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
Í
&gradients/rnn/while/Exit_6_grad/b_exitEntergradients/zeros_3*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
Í
&gradients/rnn/while/Exit_7_grad/b_exitEntergradients/zeros_4*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
Í
&gradients/rnn/while/Exit_8_grad/b_exitEntergradients/zeros_5*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:	
º
*gradients/rnn/while/Switch_2_grad/b_switchMerge&gradients/rnn/while/Exit_2_grad/b_exit1gradients/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N*
_output_shapes
: : 
Ã
*gradients/rnn/while/Switch_3_grad/b_switchMerge&gradients/rnn/while/Exit_3_grad/b_exit1gradients/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ã
*gradients/rnn/while/Switch_4_grad/b_switchMerge&gradients/rnn/while/Exit_4_grad/b_exit1gradients/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ã
*gradients/rnn/while/Switch_5_grad/b_switchMerge&gradients/rnn/while/Exit_5_grad/b_exit1gradients/rnn/while/Switch_5_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ã
*gradients/rnn/while/Switch_6_grad/b_switchMerge&gradients/rnn/while/Exit_6_grad/b_exit1gradients/rnn/while/Switch_6_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ã
*gradients/rnn/while/Switch_7_grad/b_switchMerge&gradients/rnn/while/Exit_7_grad/b_exit1gradients/rnn/while/Switch_7_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ã
*gradients/rnn/while/Switch_8_grad/b_switchMerge&gradients/rnn/while/Exit_8_grad/b_exit1gradients/rnn/while/Switch_8_grad_1/NextIteration*
T0*
N*!
_output_shapes
:	: 
Ô
'gradients/rnn/while/Merge_2_grad/SwitchSwitch*gradients/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*
_output_shapes
: : *=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
c
1gradients/rnn/while/Merge_2_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_2_grad/Switch

9gradients/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_2_grad/Switch2^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*
_output_shapes
: *=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch

;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_2_grad/Switch:12^gradients/rnn/while/Merge_2_grad/tuple/group_deps*
T0*
_output_shapes
: *=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
æ
'gradients/rnn/while/Merge_3_grad/SwitchSwitch*gradients/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
c
1gradients/rnn/while/Merge_3_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_3_grad/Switch

9gradients/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_3_grad/Switch2^gradients/rnn/while/Merge_3_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch

;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_3_grad/Switch:12^gradients/rnn/while/Merge_3_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
æ
'gradients/rnn/while/Merge_4_grad/SwitchSwitch*gradients/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
c
1gradients/rnn/while/Merge_4_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_4_grad/Switch

9gradients/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_4_grad/Switch2^gradients/rnn/while/Merge_4_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch

;gradients/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_4_grad/Switch:12^gradients/rnn/while/Merge_4_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
æ
'gradients/rnn/while/Merge_5_grad/SwitchSwitch*gradients/rnn/while/Switch_5_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
c
1gradients/rnn/while/Merge_5_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_5_grad/Switch

9gradients/rnn/while/Merge_5_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_5_grad/Switch2^gradients/rnn/while/Merge_5_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch

;gradients/rnn/while/Merge_5_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_5_grad/Switch:12^gradients/rnn/while/Merge_5_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
æ
'gradients/rnn/while/Merge_6_grad/SwitchSwitch*gradients/rnn/while/Switch_6_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_6_grad/b_switch
c
1gradients/rnn/while/Merge_6_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_6_grad/Switch

9gradients/rnn/while/Merge_6_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_6_grad/Switch2^gradients/rnn/while/Merge_6_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_6_grad/b_switch

;gradients/rnn/while/Merge_6_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_6_grad/Switch:12^gradients/rnn/while/Merge_6_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_6_grad/b_switch
æ
'gradients/rnn/while/Merge_7_grad/SwitchSwitch*gradients/rnn/while/Switch_7_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch
c
1gradients/rnn/while/Merge_7_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_7_grad/Switch

9gradients/rnn/while/Merge_7_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_7_grad/Switch2^gradients/rnn/while/Merge_7_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch

;gradients/rnn/while/Merge_7_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_7_grad/Switch:12^gradients/rnn/while/Merge_7_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch
æ
'gradients/rnn/while/Merge_8_grad/SwitchSwitch*gradients/rnn/while/Switch_8_grad/b_switchgradients/b_count_2*
T0**
_output_shapes
:	:	*=
_class3
1/loc:@gradients/rnn/while/Switch_8_grad/b_switch
c
1gradients/rnn/while/Merge_8_grad/tuple/group_depsNoOp(^gradients/rnn/while/Merge_8_grad/Switch

9gradients/rnn/while/Merge_8_grad/tuple/control_dependencyIdentity'gradients/rnn/while/Merge_8_grad/Switch2^gradients/rnn/while/Merge_8_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_8_grad/b_switch

;gradients/rnn/while/Merge_8_grad/tuple/control_dependency_1Identity)gradients/rnn/while/Merge_8_grad/Switch:12^gradients/rnn/while/Merge_8_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_8_grad/b_switch

%gradients/rnn/while/Enter_2_grad/ExitExit9gradients/rnn/while/Merge_2_grad/tuple/control_dependency*
T0*
_output_shapes
: 

%gradients/rnn/while/Enter_3_grad/ExitExit9gradients/rnn/while/Merge_3_grad/tuple/control_dependency*
T0*
_output_shapes
:	

%gradients/rnn/while/Enter_4_grad/ExitExit9gradients/rnn/while/Merge_4_grad/tuple/control_dependency*
T0*
_output_shapes
:	

%gradients/rnn/while/Enter_5_grad/ExitExit9gradients/rnn/while/Merge_5_grad/tuple/control_dependency*
T0*
_output_shapes
:	

%gradients/rnn/while/Enter_6_grad/ExitExit9gradients/rnn/while/Merge_6_grad/tuple/control_dependency*
T0*
_output_shapes
:	

%gradients/rnn/while/Enter_7_grad/ExitExit9gradients/rnn/while/Merge_7_grad/tuple/control_dependency*
T0*
_output_shapes
:	

%gradients/rnn/while/Enter_8_grad/ExitExit9gradients/rnn/while/Merge_8_grad/tuple/control_dependency*
T0*
_output_shapes
:	
õ
^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1*
_output_shapes

:: *
source	gradients*(
_class
loc:@rnn/while/dropout/mul
®
dgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterrnn/TensorArray*
is_constant(*(
_class
loc:@rnn/while/dropout/mul*
T0*
parallel_iterations *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
Ï
Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1_^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*
_output_shapes
: *(
_class
loc:@rnn/while/dropout/mul
¨
Ngradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*
_output_shapes
:	
È
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_1

Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_1
¯
Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:

Zgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Tgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterrnn/while/Identity_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
: 

Ygradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
: 
Ä
_gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterTgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ã
Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggerZ^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV24^gradients/rnn/while/dropout/div_grad/Neg/StackPopV24^gradients/rnn/while/dropout/mul_grad/Mul/StackPopV26^gradients/rnn/while/dropout/mul_grad/Mul_1/StackPopV2X^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2X^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2X^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2P^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPopV2R^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2
ä
Mgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOp<^gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1O^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3

Ugradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityNgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*
_output_shapes
:	*a
_classW
USloc:@gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
Ð
Wgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity;gradients/rnn/while/Merge_2_grad/tuple/control_dependency_1N^gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*
_output_shapes
: *=
_class3
1/loc:@gradients/rnn/while/Switch_2_grad/b_switch
å
(gradients/rnn/while/dropout/mul_grad/MulMulUgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency3gradients/rnn/while/dropout/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
¥
.gradients/rnn/while/dropout/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: **
_class 
loc:@rnn/while/dropout/Floor
Ô
.gradients/rnn/while/dropout/mul_grad/Mul/f_accStackV2.gradients/rnn/while/dropout/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name **
_class 
loc:@rnn/while/dropout/Floor
ã
.gradients/rnn/while/dropout/mul_grad/Mul/EnterEnter.gradients/rnn/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
Ù
4gradients/rnn/while/dropout/mul_grad/Mul/StackPushV2StackPushV2.gradients/rnn/while/dropout/mul_grad/Mul/Enterrnn/while/dropout/Floor^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
¾
3gradients/rnn/while/dropout/mul_grad/Mul/StackPopV2
StackPopV29gradients/rnn/while/dropout/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
ø
9gradients/rnn/while/dropout/mul_grad/Mul/StackPopV2/EnterEnter.gradients/rnn/while/dropout/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
é
*gradients/rnn/while/dropout/mul_grad/Mul_1MulUgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency5gradients/rnn/while/dropout/mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
¥
0gradients/rnn/while/dropout/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *(
_class
loc:@rnn/while/dropout/div
Ö
0gradients/rnn/while/dropout/mul_grad/Mul_1/f_accStackV20gradients/rnn/while/dropout/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *(
_class
loc:@rnn/while/dropout/div
ç
0gradients/rnn/while/dropout/mul_grad/Mul_1/EnterEnter0gradients/rnn/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
Û
6gradients/rnn/while/dropout/mul_grad/Mul_1/StackPushV2StackPushV20gradients/rnn/while/dropout/mul_grad/Mul_1/Enterrnn/while/dropout/div^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
Â
5gradients/rnn/while/dropout/mul_grad/Mul_1/StackPopV2
StackPopV2;gradients/rnn/while/dropout/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
ü
;gradients/rnn/while/dropout/mul_grad/Mul_1/StackPopV2/EnterEnter0gradients/rnn/while/dropout/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:

5gradients/rnn/while/dropout/mul_grad/tuple/group_depsNoOp)^gradients/rnn/while/dropout/mul_grad/Mul+^gradients/rnn/while/dropout/mul_grad/Mul_1

=gradients/rnn/while/dropout/mul_grad/tuple/control_dependencyIdentity(gradients/rnn/while/dropout/mul_grad/Mul6^gradients/rnn/while/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*;
_class1
/-loc:@gradients/rnn/while/dropout/mul_grad/Mul

?gradients/rnn/while/dropout/mul_grad/tuple/control_dependency_1Identity*gradients/rnn/while/dropout/mul_grad/Mul_16^gradients/rnn/while/dropout/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/dropout/mul_grad/Mul_1

*gradients/rnn/while/dropout/div_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:

,gradients/rnn/while/dropout/div_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
ê
:gradients/rnn/while/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/rnn/while/dropout/div_grad/Shape,gradients/rnn/while/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
Ô
,gradients/rnn/while/dropout/div_grad/RealDivRealDiv=gradients/rnn/while/dropout/mul_grad/tuple/control_dependency2gradients/rnn/while/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes
:	

2gradients/rnn/while/dropout/div_grad/RealDiv/ConstConst^gradients/Sub*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 
à
(gradients/rnn/while/dropout/div_grad/SumSum,gradients/rnn/while/dropout/div_grad/RealDiv:gradients/rnn/while/dropout/div_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:	*

Tidx0
Å
,gradients/rnn/while/dropout/div_grad/ReshapeReshape(gradients/rnn/while/dropout/div_grad/Sum*gradients/rnn/while/dropout/div_grad/Shape*
T0*
Tshape0*
_output_shapes
:	

(gradients/rnn/while/dropout/div_grad/NegNeg3gradients/rnn/while/dropout/div_grad/Neg/StackPopV2*
T0*
_output_shapes
:	
Á
.gradients/rnn/while/dropout/div_grad/Neg/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *F
_class<
:8loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2
ð
.gradients/rnn/while/dropout/div_grad/Neg/f_accStackV2.gradients/rnn/while/dropout/div_grad/Neg/Const*
	elem_type0*
_output_shapes
:*

stack_name *F
_class<
:8loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2
ã
.gradients/rnn/while/dropout/div_grad/Neg/EnterEnter.gradients/rnn/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
õ
4gradients/rnn/while/dropout/div_grad/Neg/StackPushV2StackPushV2.gradients/rnn/while/dropout/div_grad/Neg/Enter3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
¾
3gradients/rnn/while/dropout/div_grad/Neg/StackPopV2
StackPopV29gradients/rnn/while/dropout/div_grad/Neg/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
ø
9gradients/rnn/while/dropout/div_grad/Neg/StackPopV2/EnterEnter.gradients/rnn/while/dropout/div_grad/Neg/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
Á
.gradients/rnn/while/dropout/div_grad/RealDiv_1RealDiv(gradients/rnn/while/dropout/div_grad/Neg2gradients/rnn/while/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes
:	
Ç
.gradients/rnn/while/dropout/div_grad/RealDiv_2RealDiv.gradients/rnn/while/dropout/div_grad/RealDiv_12gradients/rnn/while/dropout/div_grad/RealDiv/Const*
T0*
_output_shapes
:	
È
(gradients/rnn/while/dropout/div_grad/mulMul=gradients/rnn/while/dropout/mul_grad/tuple/control_dependency.gradients/rnn/while/dropout/div_grad/RealDiv_2*
T0*
_output_shapes
:	
×
*gradients/rnn/while/dropout/div_grad/Sum_1Sum(gradients/rnn/while/dropout/div_grad/mul<gradients/rnn/while/dropout/div_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
Â
.gradients/rnn/while/dropout/div_grad/Reshape_1Reshape*gradients/rnn/while/dropout/div_grad/Sum_1,gradients/rnn/while/dropout/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 

5gradients/rnn/while/dropout/div_grad/tuple/group_depsNoOp-^gradients/rnn/while/dropout/div_grad/Reshape/^gradients/rnn/while/dropout/div_grad/Reshape_1

=gradients/rnn/while/dropout/div_grad/tuple/control_dependencyIdentity,gradients/rnn/while/dropout/div_grad/Reshape6^gradients/rnn/while/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
:	*?
_class5
31loc:@gradients/rnn/while/dropout/div_grad/Reshape

?gradients/rnn/while/dropout/div_grad/tuple/control_dependency_1Identity.gradients/rnn/while/dropout/div_grad/Reshape_16^gradients/rnn/while/dropout/div_grad/tuple/group_deps*
T0*
_output_shapes
: *A
_class7
53loc:@gradients/rnn/while/dropout/div_grad/Reshape_1
¼
1gradients/rnn/while/Switch_2_grad_1/NextIterationNextIterationWgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 

gradients/AddNAddN;gradients/rnn/while/Merge_8_grad/tuple/control_dependency_1=gradients/rnn/while/dropout/div_grad/tuple/control_dependency*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_8_grad/b_switch
Ú
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/MulMulgradients/AddNQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
à
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1
­
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
²
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/Enter4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
Þ
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1Mulgradients/AddNSgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:	
´
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	

gradients/AddN_1AddN;gradients/rnn/while/Merge_7_grad/tuple/control_dependency_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch
n
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_1
¸
[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch
º
]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_7_grad/b_switch
£
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/MulMul[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependencyOgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
¾
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_7

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/f_accStackV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_7

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/Enterrnn/while/Identity_7^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ö
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1Mul[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependencyQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
á
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid
®
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
³
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/Enter5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
é
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/group_depsNoOpE^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/MulG^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/control_dependencyIdentityDgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/MulR^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul

[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/control_dependency_1IdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1
©
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/MulMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependency_1Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
Þ
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh
«
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
°
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/Enter2rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1Mul]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1
·
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:	
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	
²
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
¾
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/ShapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
¼
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/SumSumPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_grad/SigmoidGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:	*

Tidx0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/ReshapeReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/SumFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
·
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Sum_1SumPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Reshape_1ReshapeFgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Sum_1Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/group_depsNoOpI^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/ReshapeK^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Reshape_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/control_dependencyIdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/ReshapeR^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Reshape

[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/control_dependency_1IdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Reshape_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
: *]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/Reshape_1
É
1gradients/rnn/while/Switch_7_grad_1/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
´
Igradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concatConcatV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_grad/TanhGradYgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes
:	*

Tidx0
¡
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ç
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concat

]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concatV^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:	*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_grad/concat
 
_gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*c
_classY
WUloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/BiasAddGrad
Õ
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMulMatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_b(*
transpose_a( *
_output_shapes
:	

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

ß
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:

æ
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat
¹
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat
«
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¾
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/Enter4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	

Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
À
]gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ø
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMulM^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMulU^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul

^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*_
_classU
SQloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
´
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes	
:
º
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
±
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ß
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Ó
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

Igradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Ggradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/modFloorModIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ConstHgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ª
Igradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
¬
Kgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ì
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/modIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ShapeKgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::

Igradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/SliceSlice\gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ConcatOffsetIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes
:	

Kgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Slice_1Slice\gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/ConcatOffset:1Kgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	
ö
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/SliceL^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Slice_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/SliceU^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Slice

^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Slice_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	*^
_classT
RPloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/Slice_1
¨
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    *
dtype0* 
_output_shapes
:

·
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

¼
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 
ù
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

³
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/AddAddRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/Switch:1^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

â
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

Ö
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

¥
gradients/AddN_2AddN;gradients/rnn/while/Merge_6_grad/tuple/control_dependency_1\gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/control_dependency*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_6_grad/b_switch
Ü
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/MulMulgradients/AddN_2Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
à
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1
­
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
²
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
à
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1Mulgradients/AddN_2Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:	
´
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
Ì
1gradients/rnn/while/Switch_8_grad_1/NextIterationNextIteration^gradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	

gradients/AddN_3AddN;gradients/rnn/while/Merge_5_grad/tuple/control_dependency_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
n
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_3
¸
[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_3T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
º
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_3T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_5_grad/b_switch
£
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/MulMul[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
¾
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_5

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_accStackV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_5

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enterrnn/while/Identity_5^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ö
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1Mul[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependencyQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
á
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid
®
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
³
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
é
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_depsNoOpE^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/MulG^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependencyIdentityDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/MulR^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul

[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1IdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1
©
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/MulMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
Þ
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh
«
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
°
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter2rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1Mul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1
·
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:	
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	
²
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
¾
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ShapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
¼
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumSumPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:	*

Tidx0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/SumFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
·
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1SumPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1ReshapeFgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Sum_1Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_depsNoOpI^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeK^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyIdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/ReshapeR^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape

[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependency_1IdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
: *]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/Reshape_1
É
1gradients/rnn/while/Switch_5_grad_1/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
´
Igradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatConcatV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_grad/TanhGradYgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes
:	*

Tidx0
¡
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ç
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat

]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concatV^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:	*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_grad/concat
 
_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*c
_classY
WUloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/BiasAddGrad
Õ
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulMatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_b(*
transpose_a( *
_output_shapes
:	

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

ß
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:

æ
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat
¹
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat
«
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¾
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	

Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
À
]gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ø
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulM^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMulU^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*_
_classU
SQloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
´
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes	
:
º
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
±
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ß
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Ó
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

Igradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Ggradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modFloorModIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConstHgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ª
Igradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
¬
Kgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ì
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/modIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ShapeKgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::

Igradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceSlice\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffsetIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes
:	

Kgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1Slice\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/ConcatOffset:1Kgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	
ö
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceL^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/SliceU^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice

^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	*^
_classT
RPloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/Slice_1
¨
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    *
dtype0* 
_output_shapes
:

·
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

¼
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 
ù
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

³
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/AddAddRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch:1^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

â
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

Ö
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

¥
gradients/AddN_4AddN;gradients/rnn/while/Merge_4_grad/tuple/control_dependency_1\gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_4_grad/b_switch
Ü
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/MulMulgradients/AddN_4Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
à
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1
­
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
²
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
à
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1Mulgradients/AddN_4Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency*
T0*
_output_shapes
:	
´
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
Ì
1gradients/rnn/while/Switch_6_grad_1/NextIterationNextIteration^gradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	

gradients/AddN_5AddN;gradients/rnn/while/Merge_3_grad/tuple/control_dependency_1Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1_grad/TanhGrad*
T0*
N*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
n
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_5
¸
[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_5T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
º
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_5T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*=
_class3
1/loc:@gradients/rnn/while/Switch_3_grad/b_switch
£
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/MulMul[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
¾
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *'
_class
loc:@rnn/while/Identity_3

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_accStackV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *'
_class
loc:@rnn/while/Identity_3

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2StackPushV2Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enterrnn/while/Identity_3^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ö
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2
StackPopV2Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
°
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPopV2/EnterEnterJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1Mul[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependencyQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
á
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid
®
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *H
_class>
<:loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
³
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
é
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_depsNoOpE^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/MulG^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependencyIdentityDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/MulR^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*W
_classM
KIloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul

[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1IdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1
©
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/MulMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0*
_output_shapes
:	
Þ
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh
«
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_accStackV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Const*
	elem_type0*
_output_shapes
:*

stack_name *E
_class;
97loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh

Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
°
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter2rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
ú
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
´
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
­
Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1Mul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1_grad/tuple/control_dependency_1Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2*
T0*
_output_shapes
:	
å
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1
´
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_accStackV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *J
_class@
><loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1
£
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¹
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	
þ
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
¸
Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterNgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ï
Sgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_depsNoOpG^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/MulI^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1

[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/MulT^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*Y
_classO
MKloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1T^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1
·
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPopV2Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency*
T0*
_output_shapes
:	
½
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPopV2[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency*
T0*
_output_shapes
:	
²
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradTanhGradQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
§
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1Const^gradients/Sub*
valueB *
dtype0*
_output_shapes
: 
¾
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ShapeHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ
¼
Dgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumSumPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradVgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:	*

Tidx0

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeReshapeDgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/SumFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape*
T0*
Tshape0*
_output_shapes
:	
·
Fgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1SumPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_grad/SigmoidGradXgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0

Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1ReshapeFgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Sum_1Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_depsNoOpI^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeK^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1

Ygradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyIdentityHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/ReshapeR^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
:	*[
_classQ
OMloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape

[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependency_1IdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1R^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/group_deps*
T0*
_output_shapes
: *]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/Reshape_1
É
1gradients/rnn/while/Switch_3_grad_1/NextIterationNextIteration[gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	
´
Igradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatConcatV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1_grad/SigmoidGradJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_grad/TanhGradYgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2_grad/SigmoidGradOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/Const*
T0*
N*
_output_shapes
:	*

Tidx0
¡
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 
ç
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:
ü
Ugradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpQ^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat

]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concatV^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes
:	*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_grad/concat
 
_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGradV^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*
_output_shapes	
:*c
_classY
WUloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/BiasAddGrad
Õ
Jgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulMatMul]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_b(*
transpose_a( *
_output_shapes
:	

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul/EnterEnter/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

ß
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1MatMulWgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_b( *
transpose_a(* 
_output_shapes
:

æ
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/ConstConst*
valueB :
ÿÿÿÿÿÿÿÿÿ*
dtype0*
_output_shapes
: *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat
¹
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*
_output_shapes
:*

stack_name *G
_class=
;9loc:@rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat
«
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*'

frame_namernn/while/while_context*
_output_shapes
:
¾
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat^gradients/Add*
T0*
swap_memory(*
_output_shapes
:	

Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0*
_output_shapes
:	
À
]gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
parallel_iterations *
is_constant(*1

frame_name#!gradients/rnn/while/while_context*
_output_shapes
:
ø
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_depsNoOpK^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulM^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityJgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMulU^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*
T0*
_output_shapes
:	*]
_classS
QOloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityLgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/group_deps*
T0* 
_output_shapes
:
*_
_classU
SQloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1

Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB*    *
dtype0*
_output_shapes	
:
´
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context*
_output_shapes	
:
º
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_1Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N*
_output_shapes
	:: 
ñ
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0*"
_output_shapes
::
±
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/AddAddSgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch:1_gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0*
_output_shapes	
:
ß
Xgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationNgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Add*
T0*
_output_shapes	
:
Ó
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/Switch*
T0*
_output_shapes	
:

Igradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Hgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0*
_output_shapes
: 

Ggradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modFloorModIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConstHgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Rank*
T0*
_output_shapes
: 
ª
Igradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeConst^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
¬
Kgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape_1Const^gradients/Sub*
valueB"      *
dtype0*
_output_shapes
:
ì
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetConcatOffsetGgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/modIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ShapeKgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape_1*
N* 
_output_shapes
::

Igradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceSlice\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffsetIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape*
Index0*
T0*
_output_shapes

:

Kgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1Slice\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependencyRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/ConcatOffset:1Kgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Shape_1*
Index0*
T0*
_output_shapes
:	
ö
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_depsNoOpJ^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceL^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1

\gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependencyIdentityIgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/SliceU^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes

:*\
_classR
PNloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice

^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1IdentityKgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1U^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/group_deps*
T0*
_output_shapes
:	*^
_classT
RPloc:@gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/Slice_1
¨
Ogradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
*    *
dtype0* 
_output_shapes
:

·
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1EnterOgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc*
T0*
parallel_iterations *
is_constant( *1

frame_name#!gradients/rnn/while/while_context* 
_output_shapes
:

¼
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2MergeQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_1Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N*"
_output_shapes
:
: 
ù
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/SwitchSwitchQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0*,
_output_shapes
:
:

³
Mgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/AddAddRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch:1^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

â
Wgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationMgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Add*
T0* 
_output_shapes
:

Ö
Qgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3ExitPgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/Switch*
T0* 
_output_shapes
:

Ì
1gradients/rnn/while/Switch_4_grad_1/NextIterationNextIteration^gradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	

w/Adagrad/Initializer/ConstConst*
valueB	*ÍÌÌ=*
dtype0*
_output_shapes
:	*
_class

loc:@w

	w/Adagrad
VariableV2*
shared_name *
_class

loc:@w*
_output_shapes
:	*
dtype0*
shape:	*
	container 
«
w/Adagrad/AssignAssign	w/Adagradw/Adagrad/Initializer/Const*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
e
w/Adagrad/readIdentity	w/Adagrad*
T0*
_output_shapes
:	*
_class

loc:@w
~
b/Adagrad/Initializer/ConstConst*
valueB*ÍÌÌ=*
dtype0*
_output_shapes
:*
_class

loc:@b

	b/Adagrad
VariableV2*
shared_name *
_class

loc:@b*
_output_shapes
:*
dtype0*
shape:*
	container 
¦
b/Adagrad/AssignAssign	b/Adagradb/Adagrad/Initializer/Const*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
`
b/Adagrad/readIdentity	b/Adagrad*
T0*
_output_shapes
:*
_class

loc:@b
Ü
Drnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/Initializer/ConstConst*
valueB
*ÍÌÌ=*
dtype0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
é
2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Ð
9rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/AssignAssign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradDrnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/Initializer/Const*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
á
7rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/readIdentity2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
Î
Brnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/Initializer/ConstConst*
valueB*ÍÌÌ=*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
Û
0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
Ã
7rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/AssignAssign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradBrnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/Initializer/Const*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
Ö
5rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/readIdentity0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*
T0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
Ü
Drnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/Initializer/ConstConst*
valueB
*ÍÌÌ=*
dtype0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
é
2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Ð
9rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/AssignAssign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradDrnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/Initializer/Const*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
á
7rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/readIdentity2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
Î
Brnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/Initializer/ConstConst*
valueB*ÍÌÌ=*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
Û
0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
Ã
7rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/AssignAssign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradBrnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/Initializer/Const*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
Ö
5rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/readIdentity0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*
T0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
Ü
Drnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/Initializer/ConstConst*
valueB
*ÍÌÌ=*
dtype0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
é
2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad
VariableV2*
shared_name *=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel* 
_output_shapes
:
*
dtype0*
shape:
*
	container 
Ð
9rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/AssignAssign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradDrnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/Initializer/Const*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
á
7rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/readIdentity2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad*
T0* 
_output_shapes
:
*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
Î
Brnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/Initializer/ConstConst*
valueB*ÍÌÌ=*
dtype0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
Û
0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad
VariableV2*
shared_name *;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias*
_output_shapes	
:*
dtype0*
shape:*
	container 
Ã
7rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/AssignAssign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradBrnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/Initializer/Const*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
Ö
5rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/readIdentity0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*
T0*
_output_shapes	
:*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
Z
Adagrad/learning_rateConst*
valueB
 *
×£;*
dtype0*
_output_shapes
: 
æ
Adagrad/update_w/ApplyAdagradApplyAdagradw	w/AdagradAdagrad/learning_rate+gradients/einsum/transpose_1_grad/transpose*
T0*
_output_shapes
:	*
use_locking( *
update_slots(*
_class

loc:@w
ã
Adagrad/update_b/ApplyAdagradApplyAdagradb	b/AdagradAdagrad/learning_rate-gradients/add_grad/tuple/control_dependency_1*
T0*
_output_shapes
:*
use_locking( *
update_slots(*
_class

loc:@b
±
FAdagrad/update_rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdagradApplyAdagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradAdagrad/learning_rateQgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0* 
_output_shapes
:
*
use_locking( *
update_slots(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
¥
DAdagrad/update_rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdagradApplyAdagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradAdagrad/learning_rateRgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes	
:*
use_locking( *
update_slots(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
±
FAdagrad/update_rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdagradApplyAdagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradAdagrad/learning_rateQgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0* 
_output_shapes
:
*
use_locking( *
update_slots(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
¥
DAdagrad/update_rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdagradApplyAdagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradAdagrad/learning_rateRgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes	
:*
use_locking( *
update_slots(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
±
FAdagrad/update_rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/ApplyAdagradApplyAdagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradAdagrad/learning_rateQgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0* 
_output_shapes
:
*
use_locking( *
update_slots(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
¥
DAdagrad/update_rnn/multi_rnn_cell/cell_2/lstm_cell/bias/ApplyAdagradApplyAdagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradAdagrad/learning_rateRgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0*
_output_shapes	
:*
use_locking( *
update_slots(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ÿ
AdagradNoOp^Adagrad/update_b/ApplyAdagradE^Adagrad/update_rnn/multi_rnn_cell/cell_0/lstm_cell/bias/ApplyAdagradG^Adagrad/update_rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/ApplyAdagradE^Adagrad/update_rnn/multi_rnn_cell/cell_1/lstm_cell/bias/ApplyAdagradG^Adagrad/update_rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/ApplyAdagradE^Adagrad/update_rnn/multi_rnn_cell/cell_2/lstm_cell/bias/ApplyAdagradG^Adagrad/update_rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/ApplyAdagrad^Adagrad/update_w/ApplyAdagrad

EDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
é
FDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2EDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/ConstGDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_1KDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

KDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 

EDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillFDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concatKDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ï
HDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_4GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_5MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillHDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/concat_1MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstIDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillHDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concatMDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillJDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstIDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillHDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concatMDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillJDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ODropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
L

rnn_1/RankConst*
value	B :*
dtype0*
_output_shapes
: 
S
rnn_1/range/startConst*
value	B :*
dtype0*
_output_shapes
: 
S
rnn_1/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
n
rnn_1/rangeRangernn_1/range/start
rnn_1/Rankrnn_1/range/delta*
_output_shapes
:*

Tidx0
f
rnn_1/concat/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
S
rnn_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 

rnn_1/concatConcatV2rnn_1/concat/values_0rnn_1/rangernn_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
{
rnn_1/transpose	Transposeprediction_xrnn_1/concat*
T0*
Tperm0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Z
rnn_1/ShapeShapernn_1/transpose*
T0*
out_type0*
_output_shapes
:
c
rnn_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
e
rnn_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
e
rnn_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

rnn_1/strided_sliceStridedSlicernn_1/Shapernn_1/strided_slice/stackrnn_1/strided_slice/stack_1rnn_1/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
U
rnn_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:
X
rnn_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:
U
rnn_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

rnn_1/concat_1ConcatV2rnn_1/Constrnn_1/Const_1rnn_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
V
rnn_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
rnn_1/zerosFillrnn_1/concat_1rnn_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0
L

rnn_1/timeConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
rnn_1/TensorArrayTensorArrayV3rnn_1/strided_slice*
dynamic_size( *
identical_element_shapes(*
element_shape:	*
_output_shapes

:: *
dtype0*1
tensor_array_namernn_1/dynamic_rnn/output_0*
clear_after_read(
ÿ
rnn_1/TensorArray_1TensorArrayV3rnn_1/strided_slice*
dynamic_size( *
identical_element_shapes(*
element_shape
:*
_output_shapes

:: *
dtype0*0
tensor_array_namernn_1/dynamic_rnn/input_0*
clear_after_read(
m
rnn_1/TensorArrayUnstack/ShapeShapernn_1/transpose*
T0*
out_type0*
_output_shapes
:
v
,rnn_1/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
x
.rnn_1/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
x
.rnn_1/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
ö
&rnn_1/TensorArrayUnstack/strided_sliceStridedSlicernn_1/TensorArrayUnstack/Shape,rnn_1/TensorArrayUnstack/strided_slice/stack.rnn_1/TensorArrayUnstack/strided_slice/stack_1.rnn_1/TensorArrayUnstack/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
f
$rnn_1/TensorArrayUnstack/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
f
$rnn_1/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Ì
rnn_1/TensorArrayUnstack/rangeRange$rnn_1/TensorArrayUnstack/range/start&rnn_1/TensorArrayUnstack/strided_slice$rnn_1/TensorArrayUnstack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
ú
@rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rnn_1/TensorArray_1rnn_1/TensorArrayUnstack/rangernn_1/transposernn_1/TensorArray_1:1*
T0*
_output_shapes
: *"
_class
loc:@rnn_1/transpose
Q
rnn_1/Maximum/xConst*
value	B :*
dtype0*
_output_shapes
: 
_
rnn_1/MaximumMaximumrnn_1/Maximum/xrnn_1/strided_slice*
T0*
_output_shapes
: 
]
rnn_1/MinimumMinimumrnn_1/strided_slicernn_1/Maximum*
T0*
_output_shapes
: 
_
rnn_1/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
³
rnn_1/while/EnterEnterrnn_1/while/iteration_counter*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
: 
¢
rnn_1/while/Enter_1Enter
rnn_1/time*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
: 
«
rnn_1/while/Enter_2Enterrnn_1/TensorArray:1*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
: 
æ
rnn_1/while/Enter_3EnterEDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
è
rnn_1/while/Enter_4EnterGDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
è
rnn_1/while/Enter_5EnterGDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
ê
rnn_1/while/Enter_6EnterIDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
è
rnn_1/while/Enter_7EnterGDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
ê
rnn_1/while/Enter_8EnterIDropoutWrapperZeroState/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1*
T0*
parallel_iterations *
is_constant( *)

frame_namernn_1/while/while_context*
_output_shapes
:	
t
rnn_1/while/MergeMergernn_1/while/Enterrnn_1/while/NextIteration*
T0*
N*
_output_shapes
: : 
z
rnn_1/while/Merge_1Mergernn_1/while/Enter_1rnn_1/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
z
rnn_1/while/Merge_2Mergernn_1/while/Enter_2rnn_1/while/NextIteration_2*
T0*
N*
_output_shapes
: : 

rnn_1/while/Merge_3Mergernn_1/while/Enter_3rnn_1/while/NextIteration_3*
T0*
N*!
_output_shapes
:	: 

rnn_1/while/Merge_4Mergernn_1/while/Enter_4rnn_1/while/NextIteration_4*
T0*
N*!
_output_shapes
:	: 

rnn_1/while/Merge_5Mergernn_1/while/Enter_5rnn_1/while/NextIteration_5*
T0*
N*!
_output_shapes
:	: 

rnn_1/while/Merge_6Mergernn_1/while/Enter_6rnn_1/while/NextIteration_6*
T0*
N*!
_output_shapes
:	: 

rnn_1/while/Merge_7Mergernn_1/while/Enter_7rnn_1/while/NextIteration_7*
T0*
N*!
_output_shapes
:	: 

rnn_1/while/Merge_8Mergernn_1/while/Enter_8rnn_1/while/NextIteration_8*
T0*
N*!
_output_shapes
:	: 
d
rnn_1/while/LessLessrnn_1/while/Mergernn_1/while/Less/Enter*
T0*
_output_shapes
: 
®
rnn_1/while/Less/EnterEnterrnn_1/strided_slice*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes
: 
j
rnn_1/while/Less_1Lessrnn_1/while/Merge_1rnn_1/while/Less_1/Enter*
T0*
_output_shapes
: 
ª
rnn_1/while/Less_1/EnterEnterrnn_1/Minimum*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes
: 
b
rnn_1/while/LogicalAnd
LogicalAndrnn_1/while/Lessrnn_1/while/Less_1*
_output_shapes
: 
P
rnn_1/while/LoopCondLoopCondrnn_1/while/LogicalAnd*
_output_shapes
: 

rnn_1/while/SwitchSwitchrnn_1/while/Mergernn_1/while/LoopCond*
T0*
_output_shapes
: : *$
_class
loc:@rnn_1/while/Merge

rnn_1/while/Switch_1Switchrnn_1/while/Merge_1rnn_1/while/LoopCond*
T0*
_output_shapes
: : *&
_class
loc:@rnn_1/while/Merge_1

rnn_1/while/Switch_2Switchrnn_1/while/Merge_2rnn_1/while/LoopCond*
T0*
_output_shapes
: : *&
_class
loc:@rnn_1/while/Merge_2
¦
rnn_1/while/Switch_3Switchrnn_1/while/Merge_3rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_3
¦
rnn_1/while/Switch_4Switchrnn_1/while/Merge_4rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_4
¦
rnn_1/while/Switch_5Switchrnn_1/while/Merge_5rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_5
¦
rnn_1/while/Switch_6Switchrnn_1/while/Merge_6rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_6
¦
rnn_1/while/Switch_7Switchrnn_1/while/Merge_7rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_7
¦
rnn_1/while/Switch_8Switchrnn_1/while/Merge_8rnn_1/while/LoopCond*
T0**
_output_shapes
:	:	*&
_class
loc:@rnn_1/while/Merge_8
W
rnn_1/while/IdentityIdentityrnn_1/while/Switch:1*
T0*
_output_shapes
: 
[
rnn_1/while/Identity_1Identityrnn_1/while/Switch_1:1*
T0*
_output_shapes
: 
[
rnn_1/while/Identity_2Identityrnn_1/while/Switch_2:1*
T0*
_output_shapes
: 
d
rnn_1/while/Identity_3Identityrnn_1/while/Switch_3:1*
T0*
_output_shapes
:	
d
rnn_1/while/Identity_4Identityrnn_1/while/Switch_4:1*
T0*
_output_shapes
:	
d
rnn_1/while/Identity_5Identityrnn_1/while/Switch_5:1*
T0*
_output_shapes
:	
d
rnn_1/while/Identity_6Identityrnn_1/while/Switch_6:1*
T0*
_output_shapes
:	
d
rnn_1/while/Identity_7Identityrnn_1/while/Switch_7:1*
T0*
_output_shapes
:	
d
rnn_1/while/Identity_8Identityrnn_1/while/Switch_8:1*
T0*
_output_shapes
:	
j
rnn_1/while/add/yConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
rnn_1/while/addAddrnn_1/while/Identityrnn_1/while/add/y*
T0*
_output_shapes
: 
Ã
rnn_1/while/TensorArrayReadV3TensorArrayReadV3#rnn_1/while/TensorArrayReadV3/Enterrnn_1/while/Identity_1%rnn_1/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes

:
¿
#rnn_1/while/TensorArrayReadV3/EnterEnterrnn_1/TensorArray_1*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes
:
ê
%rnn_1/while/TensorArrayReadV3/Enter_1Enter@rnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes
: 

;rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1/axisConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
õ
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1ConcatV2rnn_1/while/TensorArrayReadV3rnn_1/while/Identity_4;rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1/axis*
T0*
N*
_output_shapes
:	*

Tidx0

6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1MatMul6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1<rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ú
<rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1/EnterEnter/rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context* 
_output_shapes
:

ú
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1BiasAdd6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ô
=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1/EnterEnter-rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes	
:

5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const_1Const^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

?rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1/split_dimConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1Split?rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1/split_dim7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2/yConst^rnn_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ô
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2Add7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:25rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2/y*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_3Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2*
T0*
_output_shapes
:	
µ
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_3Mul7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_3rnn_1/while/Identity_3*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_4Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_2Tanh7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4Mul7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_44rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_2*
T0*
_output_shapes
:	
Î
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3Add3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_33rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4*
T0*
_output_shapes
:	
¥
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_5Sigmoid7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_3Tanh3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5Mul7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_54rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_3*
T0*
_output_shapes
:	

;rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1/axisConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1ConcatV23rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5rnn_1/while/Identity_6;rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1/axis*
T0*
N*
_output_shapes
:	*

Tidx0

6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1MatMul6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1<rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ú
<rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1/EnterEnter/rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context* 
_output_shapes
:

ú
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1BiasAdd6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ô
=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1/EnterEnter-rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes	
:

5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const_1Const^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

?rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1/split_dimConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1Split?rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1/split_dim7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2/yConst^rnn_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ô
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2Add7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:25rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2/y*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_3Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2*
T0*
_output_shapes
:	
µ
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_3Mul7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_3rnn_1/while/Identity_5*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_4Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_2Tanh7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_4Mul7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_44rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_2*
T0*
_output_shapes
:	
Î
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_3Add3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_33rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_4*
T0*
_output_shapes
:	
¥
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_5Sigmoid7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_3Tanh3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_3*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_5Mul7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_54rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_3*
T0*
_output_shapes
:	

;rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1/axisConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1ConcatV23rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_5rnn_1/while/Identity_8;rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1/axis*
T0*
N*
_output_shapes
:	*

Tidx0

6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1MatMul6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1<rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1/Enter*
T0*
transpose_b( *
transpose_a( *
_output_shapes
:	
ú
<rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1/EnterEnter/rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context* 
_output_shapes
:

ú
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1BiasAdd6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1/Enter*
T0*
data_formatNHWC*
_output_shapes
:	
ô
=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1/EnterEnter-rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read*
T0*
parallel_iterations *
is_constant(*)

frame_namernn_1/while/while_context*
_output_shapes	
:

5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Const_1Const^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

?rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1/split_dimConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 

5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1Split?rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1/split_dim7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1*
T0*@
_output_shapes.
,:	:	:	:	*
	num_split

5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2/yConst^rnn_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ô
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2Add7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:25rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2/y*
T0*
_output_shapes
:	
¡
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_3Sigmoid3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2*
T0*
_output_shapes
:	
µ
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_3Mul7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_3rnn_1/while/Identity_7*
T0*
_output_shapes
:	
£
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_4Sigmoid5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_2Tanh7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:1*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_4Mul7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_44rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_2*
T0*
_output_shapes
:	
Î
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_3Add3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_33rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_4*
T0*
_output_shapes
:	
¥
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_5Sigmoid7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:3*
T0*
_output_shapes
:	

4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_3Tanh3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_3*
T0*
_output_shapes
:	
Ó
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_5Mul7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_54rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_3*
T0*
_output_shapes
:	
y
rnn_1/while/dropout/keep_probConst^rnn_1/while/Identity*
valueB
 *ÍÌÌ=*
dtype0*
_output_shapes
: 

rnn_1/while/dropout/ShapeConst^rnn_1/while/Identity*
valueB"      *
dtype0*
_output_shapes
:

&rnn_1/while/dropout/random_uniform/minConst^rnn_1/while/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

&rnn_1/while/dropout/random_uniform/maxConst^rnn_1/while/Identity*
valueB
 *  ?*
dtype0*
_output_shapes
: 
¬
0rnn_1/while/dropout/random_uniform/RandomUniformRandomUniformrnn_1/while/dropout/Shape*
T0*
dtype0*
_output_shapes
:	*
seed2 *

seed 

&rnn_1/while/dropout/random_uniform/subSub&rnn_1/while/dropout/random_uniform/max&rnn_1/while/dropout/random_uniform/min*
T0*
_output_shapes
: 
±
&rnn_1/while/dropout/random_uniform/mulMul0rnn_1/while/dropout/random_uniform/RandomUniform&rnn_1/while/dropout/random_uniform/sub*
T0*
_output_shapes
:	
£
"rnn_1/while/dropout/random_uniformAdd&rnn_1/while/dropout/random_uniform/mul&rnn_1/while/dropout/random_uniform/min*
T0*
_output_shapes
:	

rnn_1/while/dropout/addAddrnn_1/while/dropout/keep_prob"rnn_1/while/dropout/random_uniform*
T0*
_output_shapes
:	
e
rnn_1/while/dropout/FloorFloorrnn_1/while/dropout/add*
T0*
_output_shapes
:	
 
rnn_1/while/dropout/divRealDiv3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_5rnn_1/while/dropout/keep_prob*
T0*
_output_shapes
:	
|
rnn_1/while/dropout/mulMulrnn_1/while/dropout/divrnn_1/while/dropout/Floor*
T0*
_output_shapes
:	

/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV35rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterrnn_1/while/Identity_1rnn_1/while/dropout/mulrnn_1/while/Identity_2*
T0*
_output_shapes
: **
_class 
loc:@rnn_1/while/dropout/mul
û
5rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterrnn_1/TensorArray*
is_constant(**
_class 
loc:@rnn_1/while/dropout/mul*
T0*
parallel_iterations *)

frame_namernn_1/while/while_context*
_output_shapes
:
l
rnn_1/while/add_1/yConst^rnn_1/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
f
rnn_1/while/add_1Addrnn_1/while/Identity_1rnn_1/while/add_1/y*
T0*
_output_shapes
: 
\
rnn_1/while/NextIterationNextIterationrnn_1/while/add*
T0*
_output_shapes
: 
`
rnn_1/while/NextIteration_1NextIterationrnn_1/while/add_1*
T0*
_output_shapes
: 
~
rnn_1/while/NextIteration_2NextIteration/rnn_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 

rnn_1/while/NextIteration_3NextIteration3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3*
T0*
_output_shapes
:	

rnn_1/while/NextIteration_4NextIteration3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5*
T0*
_output_shapes
:	

rnn_1/while/NextIteration_5NextIteration3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_3*
T0*
_output_shapes
:	

rnn_1/while/NextIteration_6NextIteration3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_5*
T0*
_output_shapes
:	

rnn_1/while/NextIteration_7NextIteration3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_3*
T0*
_output_shapes
:	

rnn_1/while/NextIteration_8NextIteration3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_5*
T0*
_output_shapes
:	
M
rnn_1/while/ExitExitrnn_1/while/Switch*
T0*
_output_shapes
: 
Q
rnn_1/while/Exit_1Exitrnn_1/while/Switch_1*
T0*
_output_shapes
: 
Q
rnn_1/while/Exit_2Exitrnn_1/while/Switch_2*
T0*
_output_shapes
: 
Z
rnn_1/while/Exit_3Exitrnn_1/while/Switch_3*
T0*
_output_shapes
:	
Z
rnn_1/while/Exit_4Exitrnn_1/while/Switch_4*
T0*
_output_shapes
:	
Z
rnn_1/while/Exit_5Exitrnn_1/while/Switch_5*
T0*
_output_shapes
:	
Z
rnn_1/while/Exit_6Exitrnn_1/while/Switch_6*
T0*
_output_shapes
:	
Z
rnn_1/while/Exit_7Exitrnn_1/while/Switch_7*
T0*
_output_shapes
:	
Z
rnn_1/while/Exit_8Exitrnn_1/while/Switch_8*
T0*
_output_shapes
:	
¢
(rnn_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3rnn_1/TensorArrayrnn_1/while/Exit_2*
_output_shapes
: *$
_class
loc:@rnn_1/TensorArray

"rnn_1/TensorArrayStack/range/startConst*
value	B : *
dtype0*
_output_shapes
: *$
_class
loc:@rnn_1/TensorArray

"rnn_1/TensorArrayStack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: *$
_class
loc:@rnn_1/TensorArray
î
rnn_1/TensorArrayStack/rangeRange"rnn_1/TensorArrayStack/range/start(rnn_1/TensorArrayStack/TensorArraySizeV3"rnn_1/TensorArrayStack/range/delta*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0*$
_class
loc:@rnn_1/TensorArray

*rnn_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3rnn_1/TensorArrayrnn_1/TensorArrayStack/rangernn_1/while/Exit_2*
element_shape:	*
dtype0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_class
loc:@rnn_1/TensorArray
X
rnn_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:
N
rnn_1/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
U
rnn_1/range_1/startConst*
value	B :*
dtype0*
_output_shapes
: 
U
rnn_1/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
v
rnn_1/range_1Rangernn_1/range_1/startrnn_1/Rank_1rnn_1/range_1/delta*
_output_shapes
:*

Tidx0
h
rnn_1/concat_2/values_0Const*
valueB"       *
dtype0*
_output_shapes
:
U
rnn_1/concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 

rnn_1/concat_2ConcatV2rnn_1/concat_2/values_0rnn_1/range_1rnn_1/concat_2/axis*
T0*
N*
_output_shapes
:*

Tidx0

rnn_1/transpose_1	Transpose*rnn_1/TensorArrayStack/TensorArrayGatherV3rnn_1/concat_2*
T0*
Tperm0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
l
einsum_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:

einsum_1/transpose	Transposernn_1/transpose_1einsum_1/transpose/perm*
T0*
Tperm0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
einsum_1/transpose_1/permConst*
valueB"       *
dtype0*
_output_shapes
:
{
einsum_1/transpose_1	Transposew/readeinsum_1/transpose_1/perm*
T0*
Tperm0*
_output_shapes
:	
`
einsum_1/ShapeShapeeinsum_1/transpose*
T0*
out_type0*
_output_shapes
:
f
einsum_1/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
h
einsum_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
h
einsum_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
¦
einsum_1/strided_sliceStridedSliceeinsum_1/Shapeeinsum_1/strided_slice/stackeinsum_1/strided_slice/stack_1einsum_1/strided_slice/stack_2*
Index0*
end_mask *
T0*
shrink_axis_mask*
new_axis_mask *

begin_mask *
_output_shapes
: *
ellipsis_mask 
P
einsum_1/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
\
einsum_1/mulMuleinsum_1/mul/xeinsum_1/strided_slice*
T0*
_output_shapes
: 
[
einsum_1/Reshape/shape/1Const*
value
B :*
dtype0*
_output_shapes
: 

einsum_1/Reshape/shapePackeinsum_1/muleinsum_1/Reshape/shape/1*

axis *
T0*
N*
_output_shapes
:

einsum_1/ReshapeReshapeeinsum_1/transposeeinsum_1/Reshape/shape*
T0*
Tshape0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

einsum_1/MatMulMatMuleinsum_1/Reshapeeinsum_1/transpose_1*
T0*
transpose_b( *
transpose_a( *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
\
einsum_1/Reshape_1/shape/0Const*
value	B :*
dtype0*
_output_shapes
: 
\
einsum_1/Reshape_1/shape/2Const*
value	B :*
dtype0*
_output_shapes
: 
ª
einsum_1/Reshape_1/shapePackeinsum_1/Reshape_1/shape/0einsum_1/strided_sliceeinsum_1/Reshape_1/shape/2*

axis *
T0*
N*
_output_shapes
:

einsum_1/Reshape_1Reshapeeinsum_1/MatMuleinsum_1/Reshape_1/shape*
T0*
Tshape0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
n
einsum_1/transpose_2/permConst*!
valueB"          *
dtype0*
_output_shapes
:

einsum_1/transpose_2	Transposeeinsum_1/Reshape_1einsum_1/transpose_2/perm*
T0*
Tperm0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
a
logitsAddeinsum_1/transpose_2b/read*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
R
	Softmax_1Softmaxlogits*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
W
predictions/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 

predictionsArgMax	Softmax_1predictions/dimension*
output_type0	*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

Tidx0
l
Cast_1Castpredictions*

DstT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
Truncate( *

SrcT0	
d
correct_predictionsEqualCast_1prediction_y*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ü
initNoOp^b/Adagrad/Assign	^b/Assign8^rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/Assign0^rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign:^rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/Assign2^rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign8^rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/Assign0^rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign:^rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/Assign2^rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign8^rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/Assign0^rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Assign:^rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/Assign2^rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Assign^w/Adagrad/Assign	^w/Assign

GDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_1/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save/StringJoin/inputs_1Const*<
value3B1 B+_temp_40f8c2f07a73404fa1e90967dbb161e8/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
²
save/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Ó
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*
_output_shapes
: *'
_class
loc:@save/ShardedFilename

+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
T0*
N*
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
µ
save/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save/AssignAssignbsave/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save/Assign_1Assign	b/Adagradsave/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ß
save/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ç
save/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
è
save/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ð
save/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ß
save/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ç
save/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
è
save/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ð
save/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
á
save/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
é
save/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ê
save/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ò
save/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save/Assign_14Assignwsave/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w

save/Assign_15Assign	w/Adagradsave/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w

save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard

GDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_2/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_fa557ea5b32c4f19b7d9fe3fcfb01b16/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
´
save_1/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_1/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
£
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
·
save_1/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_1/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_1/AssignAssignbsave_1/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_1/Assign_1Assign	b/Adagradsave_1/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_1/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_1/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_1/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_1/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_1/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_1/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_1/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_1/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_1/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_1/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_1/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_1/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_1/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_1/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_1/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_1/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_1/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_1/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_1/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_1/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_1/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_1/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_1/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_1/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_1/Assign_14Assignwsave_1/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_1/Assign_15Assign	w/Adagradsave_1/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard

GDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_3/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_2/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_90c5f4a14df44b41930c3ba7513a7293/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
´
save_2/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_2/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_2/ShardedFilename
£
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
·
save_2/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_2/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_2/AssignAssignbsave_2/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_2/Assign_1Assign	b/Adagradsave_2/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_2/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_2/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_2/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_2/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_2/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_2/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_2/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_2/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_2/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_2/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_2/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_2/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_2/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_2/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_2/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_2/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_2/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_2/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_2/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_2/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_2/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_2/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_2/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_2/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_2/Assign_14Assignwsave_2/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_2/Assign_15Assign	w/Adagradsave_2/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_2^save_2/Assign_3^save_2/Assign_4^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard

GDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_4/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_3/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_ab62719e2929427392ae3fc72e82e5a4/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
´
save_3/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_3/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_3/ShardedFilename
£
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(

save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
T0*
_output_shapes
: 
·
save_3/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_3/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_3/AssignAssignbsave_3/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_3/Assign_1Assign	b/Adagradsave_3/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_3/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_3/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_3/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_3/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_3/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_3/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_3/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_3/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_3/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_3/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_3/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_3/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_3/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_3/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_3/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_3/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_3/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_3/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_3/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_3/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_3/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_3/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_3/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_3/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_3/Assign_14Assignwsave_3/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_3/Assign_15Assign	w/Adagradsave_3/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_2^save_3/Assign_3^save_3/Assign_4^save_3/Assign_5^save_3/Assign_6^save_3/Assign_7^save_3/Assign_8^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard

GDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_5/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_4/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_4/StringJoin/inputs_1Const*<
value3B1 B+_temp_0cdc203cee2d406e991271c1832ff55b/part*
dtype0*
_output_shapes
: 
{
save_4/StringJoin
StringJoinsave_4/Constsave_4/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_4/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_4/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_4/ShardedFilenameShardedFilenamesave_4/StringJoinsave_4/ShardedFilename/shardsave_4/num_shards*
_output_shapes
: 
´
save_4/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_4/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_4/SaveV2SaveV2save_4/ShardedFilenamesave_4/SaveV2/tensor_namessave_4/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_4/control_dependencyIdentitysave_4/ShardedFilename^save_4/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_4/ShardedFilename
£
-save_4/MergeV2Checkpoints/checkpoint_prefixesPacksave_4/ShardedFilename^save_4/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_4/MergeV2CheckpointsMergeV2Checkpoints-save_4/MergeV2Checkpoints/checkpoint_prefixessave_4/Const*
delete_old_dirs(

save_4/IdentityIdentitysave_4/Const^save_4/MergeV2Checkpoints^save_4/control_dependency*
T0*
_output_shapes
: 
·
save_4/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_4/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_4/RestoreV2	RestoreV2save_4/Constsave_4/RestoreV2/tensor_names!save_4/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_4/AssignAssignbsave_4/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_4/Assign_1Assign	b/Adagradsave_4/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_4/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_4/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_4/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_4/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_4/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_4/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_4/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_4/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_4/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_4/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_4/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_4/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_4/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_4/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_4/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_4/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_4/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_4/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_4/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_4/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_4/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_4/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_4/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_4/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_4/Assign_14Assignwsave_4/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_4/Assign_15Assign	w/Adagradsave_4/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_4/restore_shardNoOp^save_4/Assign^save_4/Assign_1^save_4/Assign_10^save_4/Assign_11^save_4/Assign_12^save_4/Assign_13^save_4/Assign_14^save_4/Assign_15^save_4/Assign_2^save_4/Assign_3^save_4/Assign_4^save_4/Assign_5^save_4/Assign_6^save_4/Assign_7^save_4/Assign_8^save_4/Assign_9
1
save_4/restore_allNoOp^save_4/restore_shard

GDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_6/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_5/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_5/StringJoin/inputs_1Const*<
value3B1 B+_temp_2ca51f98f36c45269569691912d17c95/part*
dtype0*
_output_shapes
: 
{
save_5/StringJoin
StringJoinsave_5/Constsave_5/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_5/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_5/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_5/ShardedFilenameShardedFilenamesave_5/StringJoinsave_5/ShardedFilename/shardsave_5/num_shards*
_output_shapes
: 
´
save_5/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_5/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_5/SaveV2SaveV2save_5/ShardedFilenamesave_5/SaveV2/tensor_namessave_5/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_5/control_dependencyIdentitysave_5/ShardedFilename^save_5/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_5/ShardedFilename
£
-save_5/MergeV2Checkpoints/checkpoint_prefixesPacksave_5/ShardedFilename^save_5/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_5/MergeV2CheckpointsMergeV2Checkpoints-save_5/MergeV2Checkpoints/checkpoint_prefixessave_5/Const*
delete_old_dirs(

save_5/IdentityIdentitysave_5/Const^save_5/MergeV2Checkpoints^save_5/control_dependency*
T0*
_output_shapes
: 
·
save_5/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_5/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_5/RestoreV2	RestoreV2save_5/Constsave_5/RestoreV2/tensor_names!save_5/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_5/AssignAssignbsave_5/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_5/Assign_1Assign	b/Adagradsave_5/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_5/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_5/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_5/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_5/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_5/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_5/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_5/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_5/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_5/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_5/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_5/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_5/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_5/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_5/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_5/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_5/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_5/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_5/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_5/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_5/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_5/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_5/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_5/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_5/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_5/Assign_14Assignwsave_5/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_5/Assign_15Assign	w/Adagradsave_5/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_5/restore_shardNoOp^save_5/Assign^save_5/Assign_1^save_5/Assign_10^save_5/Assign_11^save_5/Assign_12^save_5/Assign_13^save_5/Assign_14^save_5/Assign_15^save_5/Assign_2^save_5/Assign_3^save_5/Assign_4^save_5/Assign_5^save_5/Assign_6^save_5/Assign_7^save_5/Assign_8^save_5/Assign_9
1
save_5/restore_allNoOp^save_5/restore_shard

GDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_7/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_6/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_6/StringJoin/inputs_1Const*<
value3B1 B+_temp_6ccb09d3f36a4c0e9ebb8343425fafe6/part*
dtype0*
_output_shapes
: 
{
save_6/StringJoin
StringJoinsave_6/Constsave_6/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_6/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_6/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_6/ShardedFilenameShardedFilenamesave_6/StringJoinsave_6/ShardedFilename/shardsave_6/num_shards*
_output_shapes
: 
´
save_6/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_6/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_6/SaveV2SaveV2save_6/ShardedFilenamesave_6/SaveV2/tensor_namessave_6/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_6/control_dependencyIdentitysave_6/ShardedFilename^save_6/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_6/ShardedFilename
£
-save_6/MergeV2Checkpoints/checkpoint_prefixesPacksave_6/ShardedFilename^save_6/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_6/MergeV2CheckpointsMergeV2Checkpoints-save_6/MergeV2Checkpoints/checkpoint_prefixessave_6/Const*
delete_old_dirs(

save_6/IdentityIdentitysave_6/Const^save_6/MergeV2Checkpoints^save_6/control_dependency*
T0*
_output_shapes
: 
·
save_6/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_6/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_6/RestoreV2	RestoreV2save_6/Constsave_6/RestoreV2/tensor_names!save_6/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_6/AssignAssignbsave_6/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_6/Assign_1Assign	b/Adagradsave_6/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_6/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_6/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_6/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_6/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_6/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_6/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_6/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_6/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_6/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_6/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_6/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_6/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_6/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_6/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_6/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_6/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_6/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_6/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_6/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_6/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_6/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_6/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_6/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_6/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_6/Assign_14Assignwsave_6/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_6/Assign_15Assign	w/Adagradsave_6/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_6/restore_shardNoOp^save_6/Assign^save_6/Assign_1^save_6/Assign_10^save_6/Assign_11^save_6/Assign_12^save_6/Assign_13^save_6/Assign_14^save_6/Assign_15^save_6/Assign_2^save_6/Assign_3^save_6/Assign_4^save_6/Assign_5^save_6/Assign_6^save_6/Assign_7^save_6/Assign_8^save_6/Assign_9
1
save_6/restore_allNoOp^save_6/restore_shard

GDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_8/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_7/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_7/StringJoin/inputs_1Const*<
value3B1 B+_temp_19f40c176e2d479c8060f99239f7fefe/part*
dtype0*
_output_shapes
: 
{
save_7/StringJoin
StringJoinsave_7/Constsave_7/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_7/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_7/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_7/ShardedFilenameShardedFilenamesave_7/StringJoinsave_7/ShardedFilename/shardsave_7/num_shards*
_output_shapes
: 
´
save_7/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_7/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_7/SaveV2SaveV2save_7/ShardedFilenamesave_7/SaveV2/tensor_namessave_7/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_7/control_dependencyIdentitysave_7/ShardedFilename^save_7/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_7/ShardedFilename
£
-save_7/MergeV2Checkpoints/checkpoint_prefixesPacksave_7/ShardedFilename^save_7/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_7/MergeV2CheckpointsMergeV2Checkpoints-save_7/MergeV2Checkpoints/checkpoint_prefixessave_7/Const*
delete_old_dirs(

save_7/IdentityIdentitysave_7/Const^save_7/MergeV2Checkpoints^save_7/control_dependency*
T0*
_output_shapes
: 
·
save_7/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_7/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_7/RestoreV2	RestoreV2save_7/Constsave_7/RestoreV2/tensor_names!save_7/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_7/AssignAssignbsave_7/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_7/Assign_1Assign	b/Adagradsave_7/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_7/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_7/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_7/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_7/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_7/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_7/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_7/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_7/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_7/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_7/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_7/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_7/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_7/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_7/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_7/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_7/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_7/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_7/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_7/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_7/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_7/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_7/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_7/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_7/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_7/Assign_14Assignwsave_7/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_7/Assign_15Assign	w/Adagradsave_7/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_7/restore_shardNoOp^save_7/Assign^save_7/Assign_1^save_7/Assign_10^save_7/Assign_11^save_7/Assign_12^save_7/Assign_13^save_7/Assign_14^save_7/Assign_15^save_7/Assign_2^save_7/Assign_3^save_7/Assign_4^save_7/Assign_5^save_7/Assign_6^save_7/Assign_7^save_7/Assign_8^save_7/Assign_9
1
save_7/restore_allNoOp^save_7/restore_shard

GDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

MDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ñ
HDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2GDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/ConstIDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_1MDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

MDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
¤
GDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillHDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concatMDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
÷
JDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_4IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_5ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillJDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstKDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillJDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concatODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillLDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ù
JDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstKDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

ODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
ª
IDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillJDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concatODropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ÿ
LDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
°
KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillLDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1QDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

KDropoutWrapperZeroState_9/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_8/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_8/StringJoin/inputs_1Const*<
value3B1 B+_temp_560392808d0e40b2b93bc3795bfb770a/part*
dtype0*
_output_shapes
: 
{
save_8/StringJoin
StringJoinsave_8/Constsave_8/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_8/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_8/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_8/ShardedFilenameShardedFilenamesave_8/StringJoinsave_8/ShardedFilename/shardsave_8/num_shards*
_output_shapes
: 
´
save_8/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_8/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_8/SaveV2SaveV2save_8/ShardedFilenamesave_8/SaveV2/tensor_namessave_8/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_8/control_dependencyIdentitysave_8/ShardedFilename^save_8/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_8/ShardedFilename
£
-save_8/MergeV2Checkpoints/checkpoint_prefixesPacksave_8/ShardedFilename^save_8/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_8/MergeV2CheckpointsMergeV2Checkpoints-save_8/MergeV2Checkpoints/checkpoint_prefixessave_8/Const*
delete_old_dirs(

save_8/IdentityIdentitysave_8/Const^save_8/MergeV2Checkpoints^save_8/control_dependency*
T0*
_output_shapes
: 
·
save_8/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_8/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_8/RestoreV2	RestoreV2save_8/Constsave_8/RestoreV2/tensor_names!save_8/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_8/AssignAssignbsave_8/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_8/Assign_1Assign	b/Adagradsave_8/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_8/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_8/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_8/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_8/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_8/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_8/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_8/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_8/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_8/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_8/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_8/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_8/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_8/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_8/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_8/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_8/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_8/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_8/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_8/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_8/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_8/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_8/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_8/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_8/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_8/Assign_14Assignwsave_8/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_8/Assign_15Assign	w/Adagradsave_8/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_8/restore_shardNoOp^save_8/Assign^save_8/Assign_1^save_8/Assign_10^save_8/Assign_11^save_8/Assign_12^save_8/Assign_13^save_8/Assign_14^save_8/Assign_15^save_8/Assign_2^save_8/Assign_3^save_8/Assign_4^save_8/Assign_5^save_8/Assign_6^save_8/Assign_7^save_8/Assign_8^save_8/Assign_9
1
save_8/restore_allNoOp^save_8/restore_shard

HDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_10/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
R
save_9/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_9/StringJoin/inputs_1Const*<
value3B1 B+_temp_0aded3bea710481f99becd97ea0b5fa9/part*
dtype0*
_output_shapes
: 
{
save_9/StringJoin
StringJoinsave_9/Constsave_9/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
S
save_9/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_9/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_9/ShardedFilenameShardedFilenamesave_9/StringJoinsave_9/ShardedFilename/shardsave_9/num_shards*
_output_shapes
: 
´
save_9/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_9/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
Û
save_9/SaveV2SaveV2save_9/ShardedFilenamesave_9/SaveV2/tensor_namessave_9/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_9/control_dependencyIdentitysave_9/ShardedFilename^save_9/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_9/ShardedFilename
£
-save_9/MergeV2Checkpoints/checkpoint_prefixesPacksave_9/ShardedFilename^save_9/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_9/MergeV2CheckpointsMergeV2Checkpoints-save_9/MergeV2Checkpoints/checkpoint_prefixessave_9/Const*
delete_old_dirs(

save_9/IdentityIdentitysave_9/Const^save_9/MergeV2Checkpoints^save_9/control_dependency*
T0*
_output_shapes
: 
·
save_9/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

!save_9/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ã
save_9/RestoreV2	RestoreV2save_9/Constsave_9/RestoreV2/tensor_names!save_9/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_9/AssignAssignbsave_9/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_9/Assign_1Assign	b/Adagradsave_9/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
ã
save_9/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_9/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ë
save_9/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_9/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
ì
save_9/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_9/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ô
save_9/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_9/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ã
save_9/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_9/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ë
save_9/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_9/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
ì
save_9/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_9/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ô
save_9/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_9/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
å
save_9/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_9/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
í
save_9/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_9/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
î
save_9/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_9/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ö
save_9/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_9/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_9/Assign_14Assignwsave_9/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
£
save_9/Assign_15Assign	w/Adagradsave_9/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
À
save_9/restore_shardNoOp^save_9/Assign^save_9/Assign_1^save_9/Assign_10^save_9/Assign_11^save_9/Assign_12^save_9/Assign_13^save_9/Assign_14^save_9/Assign_15^save_9/Assign_2^save_9/Assign_3^save_9/Assign_4^save_9/Assign_5^save_9/Assign_6^save_9/Assign_7^save_9/Assign_8^save_9/Assign_9
1
save_9/restore_allNoOp^save_9/restore_shard

HDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_11/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_10/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_10/StringJoin/inputs_1Const*<
value3B1 B+_temp_90f064ad573748b6b1424718bac490e7/part*
dtype0*
_output_shapes
: 
~
save_10/StringJoin
StringJoinsave_10/Constsave_10/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_10/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_10/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_10/ShardedFilenameShardedFilenamesave_10/StringJoinsave_10/ShardedFilename/shardsave_10/num_shards*
_output_shapes
: 
µ
save_10/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_10/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_10/SaveV2SaveV2save_10/ShardedFilenamesave_10/SaveV2/tensor_namessave_10/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_10/control_dependencyIdentitysave_10/ShardedFilename^save_10/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_10/ShardedFilename
¦
.save_10/MergeV2Checkpoints/checkpoint_prefixesPacksave_10/ShardedFilename^save_10/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_10/MergeV2CheckpointsMergeV2Checkpoints.save_10/MergeV2Checkpoints/checkpoint_prefixessave_10/Const*
delete_old_dirs(

save_10/IdentityIdentitysave_10/Const^save_10/MergeV2Checkpoints^save_10/control_dependency*
T0*
_output_shapes
: 
¸
save_10/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_10/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_10/RestoreV2	RestoreV2save_10/Constsave_10/RestoreV2/tensor_names"save_10/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_10/AssignAssignbsave_10/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_10/Assign_1Assign	b/Adagradsave_10/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_10/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_10/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_10/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_10/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_10/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_10/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_10/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_10/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_10/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_10/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_10/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_10/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_10/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_10/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_10/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_10/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_10/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_10/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_10/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_10/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_10/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_10/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_10/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_10/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_10/Assign_14Assignwsave_10/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_10/Assign_15Assign	w/Adagradsave_10/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_10/restore_shardNoOp^save_10/Assign^save_10/Assign_1^save_10/Assign_10^save_10/Assign_11^save_10/Assign_12^save_10/Assign_13^save_10/Assign_14^save_10/Assign_15^save_10/Assign_2^save_10/Assign_3^save_10/Assign_4^save_10/Assign_5^save_10/Assign_6^save_10/Assign_7^save_10/Assign_8^save_10/Assign_9
3
save_10/restore_allNoOp^save_10/restore_shard

HDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_12/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_11/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_11/StringJoin/inputs_1Const*<
value3B1 B+_temp_a9fc970c2ba6418ab25bc2194af2c913/part*
dtype0*
_output_shapes
: 
~
save_11/StringJoin
StringJoinsave_11/Constsave_11/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_11/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_11/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_11/ShardedFilenameShardedFilenamesave_11/StringJoinsave_11/ShardedFilename/shardsave_11/num_shards*
_output_shapes
: 
µ
save_11/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_11/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_11/SaveV2SaveV2save_11/ShardedFilenamesave_11/SaveV2/tensor_namessave_11/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_11/control_dependencyIdentitysave_11/ShardedFilename^save_11/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_11/ShardedFilename
¦
.save_11/MergeV2Checkpoints/checkpoint_prefixesPacksave_11/ShardedFilename^save_11/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_11/MergeV2CheckpointsMergeV2Checkpoints.save_11/MergeV2Checkpoints/checkpoint_prefixessave_11/Const*
delete_old_dirs(

save_11/IdentityIdentitysave_11/Const^save_11/MergeV2Checkpoints^save_11/control_dependency*
T0*
_output_shapes
: 
¸
save_11/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_11/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_11/RestoreV2	RestoreV2save_11/Constsave_11/RestoreV2/tensor_names"save_11/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_11/AssignAssignbsave_11/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_11/Assign_1Assign	b/Adagradsave_11/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_11/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_11/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_11/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_11/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_11/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_11/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_11/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_11/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_11/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_11/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_11/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_11/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_11/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_11/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_11/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_11/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_11/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_11/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_11/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_11/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_11/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_11/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_11/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_11/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_11/Assign_14Assignwsave_11/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_11/Assign_15Assign	w/Adagradsave_11/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_11/restore_shardNoOp^save_11/Assign^save_11/Assign_1^save_11/Assign_10^save_11/Assign_11^save_11/Assign_12^save_11/Assign_13^save_11/Assign_14^save_11/Assign_15^save_11/Assign_2^save_11/Assign_3^save_11/Assign_4^save_11/Assign_5^save_11/Assign_6^save_11/Assign_7^save_11/Assign_8^save_11/Assign_9
3
save_11/restore_allNoOp^save_11/restore_shard

HDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_13/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_12/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_12/StringJoin/inputs_1Const*<
value3B1 B+_temp_72732465a4924700aabb2145f575c8c9/part*
dtype0*
_output_shapes
: 
~
save_12/StringJoin
StringJoinsave_12/Constsave_12/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_12/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_12/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_12/ShardedFilenameShardedFilenamesave_12/StringJoinsave_12/ShardedFilename/shardsave_12/num_shards*
_output_shapes
: 
µ
save_12/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_12/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_12/SaveV2SaveV2save_12/ShardedFilenamesave_12/SaveV2/tensor_namessave_12/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_12/control_dependencyIdentitysave_12/ShardedFilename^save_12/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_12/ShardedFilename
¦
.save_12/MergeV2Checkpoints/checkpoint_prefixesPacksave_12/ShardedFilename^save_12/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_12/MergeV2CheckpointsMergeV2Checkpoints.save_12/MergeV2Checkpoints/checkpoint_prefixessave_12/Const*
delete_old_dirs(

save_12/IdentityIdentitysave_12/Const^save_12/MergeV2Checkpoints^save_12/control_dependency*
T0*
_output_shapes
: 
¸
save_12/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_12/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_12/RestoreV2	RestoreV2save_12/Constsave_12/RestoreV2/tensor_names"save_12/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_12/AssignAssignbsave_12/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_12/Assign_1Assign	b/Adagradsave_12/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_12/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_12/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_12/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_12/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_12/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_12/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_12/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_12/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_12/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_12/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_12/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_12/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_12/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_12/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_12/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_12/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_12/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_12/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_12/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_12/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_12/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_12/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_12/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_12/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_12/Assign_14Assignwsave_12/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_12/Assign_15Assign	w/Adagradsave_12/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_12/restore_shardNoOp^save_12/Assign^save_12/Assign_1^save_12/Assign_10^save_12/Assign_11^save_12/Assign_12^save_12/Assign_13^save_12/Assign_14^save_12/Assign_15^save_12/Assign_2^save_12/Assign_3^save_12/Assign_4^save_12/Assign_5^save_12/Assign_6^save_12/Assign_7^save_12/Assign_8^save_12/Assign_9
3
save_12/restore_allNoOp^save_12/restore_shard

HDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_14/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_13/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_13/StringJoin/inputs_1Const*<
value3B1 B+_temp_3211f2a288ab4c03a21375d643bc4e7b/part*
dtype0*
_output_shapes
: 
~
save_13/StringJoin
StringJoinsave_13/Constsave_13/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_13/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_13/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_13/ShardedFilenameShardedFilenamesave_13/StringJoinsave_13/ShardedFilename/shardsave_13/num_shards*
_output_shapes
: 
µ
save_13/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_13/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_13/SaveV2SaveV2save_13/ShardedFilenamesave_13/SaveV2/tensor_namessave_13/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_13/control_dependencyIdentitysave_13/ShardedFilename^save_13/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_13/ShardedFilename
¦
.save_13/MergeV2Checkpoints/checkpoint_prefixesPacksave_13/ShardedFilename^save_13/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_13/MergeV2CheckpointsMergeV2Checkpoints.save_13/MergeV2Checkpoints/checkpoint_prefixessave_13/Const*
delete_old_dirs(

save_13/IdentityIdentitysave_13/Const^save_13/MergeV2Checkpoints^save_13/control_dependency*
T0*
_output_shapes
: 
¸
save_13/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_13/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_13/RestoreV2	RestoreV2save_13/Constsave_13/RestoreV2/tensor_names"save_13/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_13/AssignAssignbsave_13/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_13/Assign_1Assign	b/Adagradsave_13/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_13/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_13/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_13/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_13/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_13/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_13/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_13/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_13/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_13/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_13/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_13/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_13/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_13/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_13/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_13/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_13/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_13/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_13/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_13/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_13/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_13/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_13/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_13/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_13/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_13/Assign_14Assignwsave_13/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_13/Assign_15Assign	w/Adagradsave_13/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_13/restore_shardNoOp^save_13/Assign^save_13/Assign_1^save_13/Assign_10^save_13/Assign_11^save_13/Assign_12^save_13/Assign_13^save_13/Assign_14^save_13/Assign_15^save_13/Assign_2^save_13/Assign_3^save_13/Assign_4^save_13/Assign_5^save_13/Assign_6^save_13/Assign_7^save_13/Assign_8^save_13/Assign_9
3
save_13/restore_allNoOp^save_13/restore_shard

HDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_15/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_14/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_14/StringJoin/inputs_1Const*<
value3B1 B+_temp_2733942308824bb6936181d1aadbfb36/part*
dtype0*
_output_shapes
: 
~
save_14/StringJoin
StringJoinsave_14/Constsave_14/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_14/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_14/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_14/ShardedFilenameShardedFilenamesave_14/StringJoinsave_14/ShardedFilename/shardsave_14/num_shards*
_output_shapes
: 
µ
save_14/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_14/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_14/SaveV2SaveV2save_14/ShardedFilenamesave_14/SaveV2/tensor_namessave_14/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_14/control_dependencyIdentitysave_14/ShardedFilename^save_14/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_14/ShardedFilename
¦
.save_14/MergeV2Checkpoints/checkpoint_prefixesPacksave_14/ShardedFilename^save_14/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_14/MergeV2CheckpointsMergeV2Checkpoints.save_14/MergeV2Checkpoints/checkpoint_prefixessave_14/Const*
delete_old_dirs(

save_14/IdentityIdentitysave_14/Const^save_14/MergeV2Checkpoints^save_14/control_dependency*
T0*
_output_shapes
: 
¸
save_14/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_14/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_14/RestoreV2	RestoreV2save_14/Constsave_14/RestoreV2/tensor_names"save_14/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_14/AssignAssignbsave_14/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_14/Assign_1Assign	b/Adagradsave_14/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_14/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_14/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_14/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_14/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_14/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_14/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_14/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_14/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_14/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_14/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_14/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_14/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_14/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_14/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_14/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_14/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_14/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_14/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_14/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_14/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_14/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_14/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_14/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_14/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_14/Assign_14Assignwsave_14/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_14/Assign_15Assign	w/Adagradsave_14/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_14/restore_shardNoOp^save_14/Assign^save_14/Assign_1^save_14/Assign_10^save_14/Assign_11^save_14/Assign_12^save_14/Assign_13^save_14/Assign_14^save_14/Assign_15^save_14/Assign_2^save_14/Assign_3^save_14/Assign_4^save_14/Assign_5^save_14/Assign_6^save_14/Assign_7^save_14/Assign_8^save_14/Assign_9
3
save_14/restore_allNoOp^save_14/restore_shard

HDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_16/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_15/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_15/StringJoin/inputs_1Const*<
value3B1 B+_temp_12b16b460bf74ec1abd89c0da7e88035/part*
dtype0*
_output_shapes
: 
~
save_15/StringJoin
StringJoinsave_15/Constsave_15/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_15/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_15/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_15/ShardedFilenameShardedFilenamesave_15/StringJoinsave_15/ShardedFilename/shardsave_15/num_shards*
_output_shapes
: 
µ
save_15/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_15/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_15/SaveV2SaveV2save_15/ShardedFilenamesave_15/SaveV2/tensor_namessave_15/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_15/control_dependencyIdentitysave_15/ShardedFilename^save_15/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_15/ShardedFilename
¦
.save_15/MergeV2Checkpoints/checkpoint_prefixesPacksave_15/ShardedFilename^save_15/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_15/MergeV2CheckpointsMergeV2Checkpoints.save_15/MergeV2Checkpoints/checkpoint_prefixessave_15/Const*
delete_old_dirs(

save_15/IdentityIdentitysave_15/Const^save_15/MergeV2Checkpoints^save_15/control_dependency*
T0*
_output_shapes
: 
¸
save_15/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_15/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_15/RestoreV2	RestoreV2save_15/Constsave_15/RestoreV2/tensor_names"save_15/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_15/AssignAssignbsave_15/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_15/Assign_1Assign	b/Adagradsave_15/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_15/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_15/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_15/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_15/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_15/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_15/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_15/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_15/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_15/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_15/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_15/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_15/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_15/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_15/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_15/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_15/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_15/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_15/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_15/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_15/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_15/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_15/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_15/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_15/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_15/Assign_14Assignwsave_15/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_15/Assign_15Assign	w/Adagradsave_15/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_15/restore_shardNoOp^save_15/Assign^save_15/Assign_1^save_15/Assign_10^save_15/Assign_11^save_15/Assign_12^save_15/Assign_13^save_15/Assign_14^save_15/Assign_15^save_15/Assign_2^save_15/Assign_3^save_15/Assign_4^save_15/Assign_5^save_15/Assign_6^save_15/Assign_7^save_15/Assign_8^save_15/Assign_9
3
save_15/restore_allNoOp^save_15/restore_shard

HDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_17/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_16/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_16/StringJoin/inputs_1Const*<
value3B1 B+_temp_b9e8b50350f548afb94c6d74375c78ad/part*
dtype0*
_output_shapes
: 
~
save_16/StringJoin
StringJoinsave_16/Constsave_16/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_16/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_16/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_16/ShardedFilenameShardedFilenamesave_16/StringJoinsave_16/ShardedFilename/shardsave_16/num_shards*
_output_shapes
: 
µ
save_16/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_16/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_16/SaveV2SaveV2save_16/ShardedFilenamesave_16/SaveV2/tensor_namessave_16/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_16/control_dependencyIdentitysave_16/ShardedFilename^save_16/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_16/ShardedFilename
¦
.save_16/MergeV2Checkpoints/checkpoint_prefixesPacksave_16/ShardedFilename^save_16/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_16/MergeV2CheckpointsMergeV2Checkpoints.save_16/MergeV2Checkpoints/checkpoint_prefixessave_16/Const*
delete_old_dirs(

save_16/IdentityIdentitysave_16/Const^save_16/MergeV2Checkpoints^save_16/control_dependency*
T0*
_output_shapes
: 
¸
save_16/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_16/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_16/RestoreV2	RestoreV2save_16/Constsave_16/RestoreV2/tensor_names"save_16/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_16/AssignAssignbsave_16/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_16/Assign_1Assign	b/Adagradsave_16/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_16/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_16/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_16/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_16/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_16/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_16/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_16/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_16/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_16/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_16/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_16/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_16/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_16/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_16/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_16/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_16/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_16/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_16/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_16/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_16/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_16/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_16/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_16/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_16/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_16/Assign_14Assignwsave_16/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_16/Assign_15Assign	w/Adagradsave_16/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_16/restore_shardNoOp^save_16/Assign^save_16/Assign_1^save_16/Assign_10^save_16/Assign_11^save_16/Assign_12^save_16/Assign_13^save_16/Assign_14^save_16/Assign_15^save_16/Assign_2^save_16/Assign_3^save_16/Assign_4^save_16/Assign_5^save_16/Assign_6^save_16/Assign_7^save_16/Assign_8^save_16/Assign_9
3
save_16/restore_allNoOp^save_16/restore_shard

HDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/ConstConst*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

NDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
õ
IDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concatConcatV2HDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/ConstJDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_1NDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

NDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
§
HDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zerosFillIDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concatNDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zeros/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
û
KDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat_1ConcatV2JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_4JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_5PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1FillKDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/concat_1PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concatConcatV2JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/ConstLDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_1PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zerosFillKDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concatPDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1ConcatV2LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_4LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_5RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1FillMDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/concat_1RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_1/Const_7Const*
valueB:*
dtype0*
_output_shapes
:

JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstConst*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1Const*
valueB:*
dtype0*
_output_shapes
:

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ý
KDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concatConcatV2JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/ConstLDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_1PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0

PDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
­
JDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zerosFillKDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concatPDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_2Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_3Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5Const*
valueB:*
dtype0*
_output_shapes
:

RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

MDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1ConcatV2LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_4LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_5RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0

RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
³
LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1FillMDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/concat_1RDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/zeros_1/Const*
T0*
_output_shapes
:	*

index_type0

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_6Const*
valueB:*
dtype0*
_output_shapes
:

LDropoutWrapperZeroState_18/MultiRNNCellZeroState/LSTMCellZeroState_2/Const_7Const*
valueB:*
dtype0*
_output_shapes
:
S
save_17/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 

save_17/StringJoin/inputs_1Const*<
value3B1 B+_temp_ef7f9ca9ba7743e4aa0722c54a22b873/part*
dtype0*
_output_shapes
: 
~
save_17/StringJoin
StringJoinsave_17/Constsave_17/StringJoin/inputs_1*
N*
	separator *
_output_shapes
: 
T
save_17/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
_
save_17/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 

save_17/ShardedFilenameShardedFilenamesave_17/StringJoinsave_17/ShardedFilename/shardsave_17/num_shards*
_output_shapes
: 
µ
save_17/SaveV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

save_17/SaveV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ß
save_17/SaveV2SaveV2save_17/ShardedFilenamesave_17/SaveV2/tensor_namessave_17/SaveV2/shape_and_slicesb	b/Adagrad(rnn/multi_rnn_cell/cell_0/lstm_cell/bias0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_0/lstm_cell/kernel2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_1/lstm_cell/bias0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_1/lstm_cell/kernel2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad(rnn/multi_rnn_cell/cell_2/lstm_cell/bias0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad*rnn/multi_rnn_cell/cell_2/lstm_cell/kernel2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradw	w/Adagrad*
dtypes
2

save_17/control_dependencyIdentitysave_17/ShardedFilename^save_17/SaveV2*
T0*
_output_shapes
: **
_class 
loc:@save_17/ShardedFilename
¦
.save_17/MergeV2Checkpoints/checkpoint_prefixesPacksave_17/ShardedFilename^save_17/control_dependency*

axis *
T0*
N*
_output_shapes
:

save_17/MergeV2CheckpointsMergeV2Checkpoints.save_17/MergeV2Checkpoints/checkpoint_prefixessave_17/Const*
delete_old_dirs(

save_17/IdentityIdentitysave_17/Const^save_17/MergeV2Checkpoints^save_17/control_dependency*
T0*
_output_shapes
: 
¸
save_17/RestoreV2/tensor_namesConst*å
valueÛBØBbB	b/AdagradB(rnn/multi_rnn_cell/cell_0/lstm_cell/biasB0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_1/lstm_cell/biasB0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/AdagradB(rnn/multi_rnn_cell/cell_2/lstm_cell/biasB0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/AdagradB*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelB2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/AdagradBwB	w/Adagrad*
dtype0*
_output_shapes
:

"save_17/RestoreV2/shape_and_slicesConst*3
value*B(B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
ç
save_17/RestoreV2	RestoreV2save_17/Constsave_17/RestoreV2/tensor_names"save_17/RestoreV2/shape_and_slices*
dtypes
2*T
_output_shapesB
@::::::::::::::::

save_17/AssignAssignbsave_17/RestoreV2*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b

save_17/Assign_1Assign	b/Adagradsave_17/RestoreV2:1*
T0*
_output_shapes
:*
use_locking(*
validate_shape(*
_class

loc:@b
å
save_17/Assign_2Assign(rnn/multi_rnn_cell/cell_0/lstm_cell/biassave_17/RestoreV2:2*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
í
save_17/Assign_3Assign0rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagradsave_17/RestoreV2:3*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/bias
î
save_17/Assign_4Assign*rnn/multi_rnn_cell/cell_0/lstm_cell/kernelsave_17/RestoreV2:4*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
ö
save_17/Assign_5Assign2rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagradsave_17/RestoreV2:5*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_0/lstm_cell/kernel
å
save_17/Assign_6Assign(rnn/multi_rnn_cell/cell_1/lstm_cell/biassave_17/RestoreV2:6*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
í
save_17/Assign_7Assign0rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagradsave_17/RestoreV2:7*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/bias
î
save_17/Assign_8Assign*rnn/multi_rnn_cell/cell_1/lstm_cell/kernelsave_17/RestoreV2:8*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ö
save_17/Assign_9Assign2rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagradsave_17/RestoreV2:9*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_1/lstm_cell/kernel
ç
save_17/Assign_10Assign(rnn/multi_rnn_cell/cell_2/lstm_cell/biassave_17/RestoreV2:10*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ï
save_17/Assign_11Assign0rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagradsave_17/RestoreV2:11*
T0*
_output_shapes	
:*
use_locking(*
validate_shape(*;
_class1
/-loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/bias
ð
save_17/Assign_12Assign*rnn/multi_rnn_cell/cell_2/lstm_cell/kernelsave_17/RestoreV2:12*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel
ø
save_17/Assign_13Assign2rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagradsave_17/RestoreV2:13*
T0* 
_output_shapes
:
*
use_locking(*
validate_shape(*=
_class3
1/loc:@rnn/multi_rnn_cell/cell_2/lstm_cell/kernel

save_17/Assign_14Assignwsave_17/RestoreV2:14*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
¥
save_17/Assign_15Assign	w/Adagradsave_17/RestoreV2:15*
T0*
_output_shapes
:	*
use_locking(*
validate_shape(*
_class

loc:@w
Ñ
save_17/restore_shardNoOp^save_17/Assign^save_17/Assign_1^save_17/Assign_10^save_17/Assign_11^save_17/Assign_12^save_17/Assign_13^save_17/Assign_14^save_17/Assign_15^save_17/Assign_2^save_17/Assign_3^save_17/Assign_4^save_17/Assign_5^save_17/Assign_6^save_17/Assign_7^save_17/Assign_8^save_17/Assign_9
3
save_17/restore_allNoOp^save_17/restore_shard "E
save_17/Const:0save_17/Identity:0save_17/restore_all (5 @F8"ÃÔ
while_context°Ô¬Ô
»
rnn/while/while_context  *rnn/while/LoopCond:02rnn/while/Merge:0:rnn/while/Identity:0Brnn/while/Exit:0Brnn/while/Exit_1:0Brnn/while/Exit_2:0Brnn/while/Exit_3:0Brnn/while/Exit_4:0Brnn/while/Exit_5:0Brnn/while/Exit_6:0Brnn/while/Exit_7:0Brnn/while/Exit_8:0Bgradients/f_count_2:0J©
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0
\gradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2:0
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
0gradients/rnn/while/dropout/div_grad/Neg/Enter:0
6gradients/rnn/while/dropout/div_grad/Neg/StackPushV2:0
0gradients/rnn/while/dropout/div_grad/Neg/f_acc:0
0gradients/rnn/while/dropout/mul_grad/Mul/Enter:0
6gradients/rnn/while/dropout/mul_grad/Mul/StackPushV2:0
0gradients/rnn/while/dropout/mul_grad/Mul/f_acc:0
2gradients/rnn/while/dropout/mul_grad/Mul_1/Enter:0
8gradients/rnn/while/dropout/mul_grad/Mul_1/StackPushV2:0
2gradients/rnn/while/dropout/mul_grad/Mul_1/f_acc:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/StackPushV2:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/StackPushV2:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/Enter:0
Zgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/StackPushV2:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/f_acc:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/Enter:0
Vgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/StackPushV2:0
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/f_acc:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/Enter:0
Rgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/StackPushV2:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/f_acc:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/Enter:0
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/StackPushV2:0
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/f_acc:0
rnn/Minimum:0
rnn/TensorArray:0
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn/TensorArray_1:0
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:0
rnn/strided_slice:0
rnn/while/Enter:0
rnn/while/Enter_1:0
rnn/while/Enter_2:0
rnn/while/Enter_3:0
rnn/while/Enter_4:0
rnn/while/Enter_5:0
rnn/while/Enter_6:0
rnn/while/Enter_7:0
rnn/while/Enter_8:0
rnn/while/Exit:0
rnn/while/Exit_1:0
rnn/while/Exit_2:0
rnn/while/Exit_3:0
rnn/while/Exit_4:0
rnn/while/Exit_5:0
rnn/while/Exit_6:0
rnn/while/Exit_7:0
rnn/while/Exit_8:0
rnn/while/Identity:0
rnn/while/Identity_1:0
rnn/while/Identity_2:0
rnn/while/Identity_3:0
rnn/while/Identity_4:0
rnn/while/Identity_5:0
rnn/while/Identity_6:0
rnn/while/Identity_7:0
rnn/while/Identity_8:0
rnn/while/Less/Enter:0
rnn/while/Less:0
rnn/while/Less_1/Enter:0
rnn/while/Less_1:0
rnn/while/LogicalAnd:0
rnn/while/LoopCond:0
rnn/while/Merge:0
rnn/while/Merge:1
rnn/while/Merge_1:0
rnn/while/Merge_1:1
rnn/while/Merge_2:0
rnn/while/Merge_2:1
rnn/while/Merge_3:0
rnn/while/Merge_3:1
rnn/while/Merge_4:0
rnn/while/Merge_4:1
rnn/while/Merge_5:0
rnn/while/Merge_5:1
rnn/while/Merge_6:0
rnn/while/Merge_6:1
rnn/while/Merge_7:0
rnn/while/Merge_7:1
rnn/while/Merge_8:0
rnn/while/Merge_8:1
rnn/while/NextIteration:0
rnn/while/NextIteration_1:0
rnn/while/NextIteration_2:0
rnn/while/NextIteration_3:0
rnn/while/NextIteration_4:0
rnn/while/NextIteration_5:0
rnn/while/NextIteration_6:0
rnn/while/NextIteration_7:0
rnn/while/NextIteration_8:0
rnn/while/Switch:0
rnn/while/Switch:1
rnn/while/Switch_1:0
rnn/while/Switch_1:1
rnn/while/Switch_2:0
rnn/while/Switch_2:1
rnn/while/Switch_3:0
rnn/while/Switch_3:1
rnn/while/Switch_4:0
rnn/while/Switch_4:1
rnn/while/Switch_5:0
rnn/while/Switch_5:1
rnn/while/Switch_6:0
rnn/while/Switch_6:1
rnn/while/Switch_7:0
rnn/while/Switch_7:1
rnn/while/Switch_8:0
rnn/while/Switch_8:1
#rnn/while/TensorArrayReadV3/Enter:0
%rnn/while/TensorArrayReadV3/Enter_1:0
rnn/while/TensorArrayReadV3:0
5rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/rnn/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn/while/add/y:0
rnn/while/add:0
rnn/while/add_1/y:0
rnn/while/add_1:0
rnn/while/dropout/Floor:0
rnn/while/dropout/Shape:0
rnn/while/dropout/add:0
rnn/while/dropout/div:0
rnn/while/dropout/keep_prob:0
rnn/while/dropout/mul:0
0rnn/while/dropout/random_uniform/RandomUniform:0
&rnn/while/dropout/random_uniform/max:0
&rnn/while/dropout/random_uniform/min:0
&rnn/while/dropout/random_uniform/mul:0
&rnn/while/dropout/random_uniform/sub:0
"rnn/while/dropout/random_uniform:0
=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const:0
<rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_1:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_2:0
4rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh:0
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_1:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add/y:0
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_1:0
;rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat/axis:0
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat:0
3rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2:0
?rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split/split_dim:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:1
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:2
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split:3
=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const:0
<rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0
6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_1:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_2:0
4rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh:0
6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_1:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add/y:0
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_1:0
;rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat/axis:0
6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat:0
3rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2:0
?rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split/split_dim:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:1
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:2
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split:3
=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Const:0
<rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter:0
6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_1:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_2:0
4rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh:0
6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_1:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add/y:0
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_1:0
;rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat/axis:0
6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat:0
3rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2:0
?rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split/split_dim:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:1
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:2
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split:3:
rnn/TensorArray_1:0#rnn/while/TensorArrayReadV3/Enter:0q
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0<rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul/Enter:0-
rnn/strided_slice:0rnn/while/Less/Enter:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul_1/Enter:0°
Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0Vgradients/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul_1/Enter:0i
@rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%rnn/while/TensorArrayReadV3/Enter_1:0d
0gradients/rnn/while/dropout/div_grad/Neg/f_acc:00gradients/rnn/while/dropout/div_grad/Neg/Enter:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul/Enter:0
Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/f_acc:0Lgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_grad/Mul/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul_1/Enter:0q
1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:0<rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul/Enter:0d
0gradients/rnn/while/dropout/mul_grad/Mul/f_acc:00gradients/rnn/while/dropout/mul_grad/Mul/Enter:0¬
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_grad/MatMul_1/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_2_grad/Mul/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul_1/Enter:0)
rnn/Minimum:0rnn/while/Less_1/Enter:0q
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0<rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_2_grad/Mul_1/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul/Enter:0p
/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:0=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_grad/Mul_1/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_1_grad/Mul_1/Enter:0J
rnn/TensorArray:05rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0p
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_1_grad/Mul/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_grad/Mul_1/Enter:0¬
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_grad/MatMul_1/Enter:0¬
Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/f_acc:0Tgradients/rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_grad/MatMul_1/Enter:0¤
Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/f_acc:0Pgradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_2_grad/Mul_1/Enter:0p
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd/Enter:0h
2gradients/rnn/while/dropout/mul_grad/Mul_1/f_acc:02gradients/rnn/while/dropout/mul_grad/Mul_1/Enter:0 
Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/f_acc:0Ngradients/rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_1_grad/Mul/Enter:0Rrnn/while/Enter:0Rrnn/while/Enter_1:0Rrnn/while/Enter_2:0Rrnn/while/Enter_3:0Rrnn/while/Enter_4:0Rrnn/while/Enter_5:0Rrnn/while/Enter_6:0Rrnn/while/Enter_7:0Rrnn/while/Enter_8:0Rgradients/f_count_1:0Zrnn/strided_slice:0
êC
rnn_1/while/while_context  *rnn_1/while/LoopCond:02rnn_1/while/Merge:0:rnn_1/while/Identity:0Brnn_1/while/Exit:0Brnn_1/while/Exit_1:0Brnn_1/while/Exit_2:0Brnn_1/while/Exit_3:0Brnn_1/while/Exit_4:0Brnn_1/while/Exit_5:0Brnn_1/while/Exit_6:0Brnn_1/while/Exit_7:0Brnn_1/while/Exit_8:0JÙ?
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0
/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:0
1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:0
?rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1/Enter:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Const_1:0
>rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1/Enter:0
8rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_3:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_4:0
9rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Sigmoid_5:0
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_2:0
6rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/Tanh_3:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2/y:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_2:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/add_3:0
=rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1/axis:0
8rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/concat_1:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_3:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_4:0
5rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/mul_5:0
Arnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1/split_dim:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:0
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:1
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:2
7rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/split_1:3
?rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1/Enter:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Const_1:0
>rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1/Enter:0
8rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_3:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_4:0
9rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Sigmoid_5:0
6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_2:0
6rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/Tanh_3:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2/y:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_2:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/add_3:0
=rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1/axis:0
8rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/concat_1:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_3:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_4:0
5rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/mul_5:0
Arnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1/split_dim:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:0
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:1
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:2
7rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/split_1:3
?rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1/Enter:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Const_1:0
>rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1/Enter:0
8rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_3:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_4:0
9rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Sigmoid_5:0
6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_2:0
6rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/Tanh_3:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2/y:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_2:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/add_3:0
=rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1/axis:0
8rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/concat_1:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_3:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_4:0
5rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/mul_5:0
Arnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1/split_dim:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:0
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:1
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:2
7rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/split_1:3
rnn_1/Minimum:0
rnn_1/TensorArray:0
Brnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
rnn_1/TensorArray_1:0
rnn_1/strided_slice:0
rnn_1/while/Enter:0
rnn_1/while/Enter_1:0
rnn_1/while/Enter_2:0
rnn_1/while/Enter_3:0
rnn_1/while/Enter_4:0
rnn_1/while/Enter_5:0
rnn_1/while/Enter_6:0
rnn_1/while/Enter_7:0
rnn_1/while/Enter_8:0
rnn_1/while/Exit:0
rnn_1/while/Exit_1:0
rnn_1/while/Exit_2:0
rnn_1/while/Exit_3:0
rnn_1/while/Exit_4:0
rnn_1/while/Exit_5:0
rnn_1/while/Exit_6:0
rnn_1/while/Exit_7:0
rnn_1/while/Exit_8:0
rnn_1/while/Identity:0
rnn_1/while/Identity_1:0
rnn_1/while/Identity_2:0
rnn_1/while/Identity_3:0
rnn_1/while/Identity_4:0
rnn_1/while/Identity_5:0
rnn_1/while/Identity_6:0
rnn_1/while/Identity_7:0
rnn_1/while/Identity_8:0
rnn_1/while/Less/Enter:0
rnn_1/while/Less:0
rnn_1/while/Less_1/Enter:0
rnn_1/while/Less_1:0
rnn_1/while/LogicalAnd:0
rnn_1/while/LoopCond:0
rnn_1/while/Merge:0
rnn_1/while/Merge:1
rnn_1/while/Merge_1:0
rnn_1/while/Merge_1:1
rnn_1/while/Merge_2:0
rnn_1/while/Merge_2:1
rnn_1/while/Merge_3:0
rnn_1/while/Merge_3:1
rnn_1/while/Merge_4:0
rnn_1/while/Merge_4:1
rnn_1/while/Merge_5:0
rnn_1/while/Merge_5:1
rnn_1/while/Merge_6:0
rnn_1/while/Merge_6:1
rnn_1/while/Merge_7:0
rnn_1/while/Merge_7:1
rnn_1/while/Merge_8:0
rnn_1/while/Merge_8:1
rnn_1/while/NextIteration:0
rnn_1/while/NextIteration_1:0
rnn_1/while/NextIteration_2:0
rnn_1/while/NextIteration_3:0
rnn_1/while/NextIteration_4:0
rnn_1/while/NextIteration_5:0
rnn_1/while/NextIteration_6:0
rnn_1/while/NextIteration_7:0
rnn_1/while/NextIteration_8:0
rnn_1/while/Switch:0
rnn_1/while/Switch:1
rnn_1/while/Switch_1:0
rnn_1/while/Switch_1:1
rnn_1/while/Switch_2:0
rnn_1/while/Switch_2:1
rnn_1/while/Switch_3:0
rnn_1/while/Switch_3:1
rnn_1/while/Switch_4:0
rnn_1/while/Switch_4:1
rnn_1/while/Switch_5:0
rnn_1/while/Switch_5:1
rnn_1/while/Switch_6:0
rnn_1/while/Switch_6:1
rnn_1/while/Switch_7:0
rnn_1/while/Switch_7:1
rnn_1/while/Switch_8:0
rnn_1/while/Switch_8:1
%rnn_1/while/TensorArrayReadV3/Enter:0
'rnn_1/while/TensorArrayReadV3/Enter_1:0
rnn_1/while/TensorArrayReadV3:0
7rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
1rnn_1/while/TensorArrayWrite/TensorArrayWriteV3:0
rnn_1/while/add/y:0
rnn_1/while/add:0
rnn_1/while/add_1/y:0
rnn_1/while/add_1:0
rnn_1/while/dropout/Floor:0
rnn_1/while/dropout/Shape:0
rnn_1/while/dropout/add:0
rnn_1/while/dropout/div:0
rnn_1/while/dropout/keep_prob:0
rnn_1/while/dropout/mul:0
2rnn_1/while/dropout/random_uniform/RandomUniform:0
(rnn_1/while/dropout/random_uniform/max:0
(rnn_1/while/dropout/random_uniform/min:0
(rnn_1/while/dropout/random_uniform/mul:0
(rnn_1/while/dropout/random_uniform/sub:0
$rnn_1/while/dropout/random_uniform:0r
/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:0?rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/BiasAdd_1/Enter:0s
1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:0>rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/MatMul_1/Enter:0s
1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:0>rnn/while/rnn/multi_rnn_cell/cell_0/lstm_cell/MatMul_1/Enter:0s
1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:0>rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/MatMul_1/Enter:0>
rnn_1/TensorArray_1:0%rnn_1/while/TensorArrayReadV3/Enter:0m
Brnn_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0'rnn_1/while/TensorArrayReadV3/Enter_1:0r
/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:0?rnn/while/rnn/multi_rnn_cell/cell_2/lstm_cell/BiasAdd_1/Enter:0N
rnn_1/TensorArray:07rnn_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0r
/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:0?rnn/while/rnn/multi_rnn_cell/cell_1/lstm_cell/BiasAdd_1/Enter:01
rnn_1/strided_slice:0rnn_1/while/Less/Enter:0-
rnn_1/Minimum:0rnn_1/while/Less_1/Enter:0Rrnn_1/while/Enter:0Rrnn_1/while/Enter_1:0Rrnn_1/while/Enter_2:0Rrnn_1/while/Enter_3:0Rrnn_1/while/Enter_4:0Rrnn_1/while/Enter_5:0Rrnn_1/while/Enter_6:0Rrnn_1/while/Enter_7:0Rrnn_1/while/Enter_8:0Zrnn_1/strided_slice:0"
trainable_variablesü
ù

-
w:0w/Assignw/read:02random_uniform:08
/
b:0b/Assignb/read:02random_uniform_1:08
ß
,rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
ß
,rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
ß
,rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Initializer/zeros:08"
train_op
	
Adagrad"à
	variablesÒÏ
-
w:0w/Assignw/read:02random_uniform:08
/
b:0b/Assignb/read:02random_uniform_1:08
ß
,rnn/multi_rnn_cell/cell_0/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_0/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_0/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Initializer/zeros:08
ß
,rnn/multi_rnn_cell/cell_1/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_1/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_1/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Initializer/zeros:08
ß
,rnn/multi_rnn_cell/cell_2/lstm_cell/kernel:01rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Assign1rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/read:02Grnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Initializer/random_uniform:08
Î
*rnn/multi_rnn_cell/cell_2/lstm_cell/bias:0/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Assign/rnn/multi_rnn_cell/cell_2/lstm_cell/bias/read:02<rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Initializer/zeros:08
P
w/Adagrad:0w/Adagrad/Assignw/Adagrad/read:02w/Adagrad/Initializer/Const:0
P
b/Adagrad:0b/Adagrad/Assignb/Adagrad/read:02b/Adagrad/Initializer/Const:0
ô
4rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad:09rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/Assign9rnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/read:02Frnn/multi_rnn_cell/cell_0/lstm_cell/kernel/Adagrad/Initializer/Const:0
ì
2rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad:07rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/Assign7rnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/read:02Drnn/multi_rnn_cell/cell_0/lstm_cell/bias/Adagrad/Initializer/Const:0
ô
4rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad:09rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/Assign9rnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/read:02Frnn/multi_rnn_cell/cell_1/lstm_cell/kernel/Adagrad/Initializer/Const:0
ì
2rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad:07rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/Assign7rnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/read:02Drnn/multi_rnn_cell/cell_1/lstm_cell/bias/Adagrad/Initializer/Const:0
ô
4rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad:09rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/Assign9rnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/read:02Frnn/multi_rnn_cell/cell_2/lstm_cell/kernel/Adagrad/Initializer/Const:0
ì
2rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad:07rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/Assign7rnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/read:02Drnn/multi_rnn_cell/cell_2/lstm_cell/bias/Adagrad/Initializer/Const:0*Ë
serving_default·
5
prediction_y%
prediction_y:0ÿÿÿÿÿÿÿÿÿ
9
prediction_x)
prediction_x:0ÿÿÿÿÿÿÿÿÿC
correct_predictions,
correct_predictions:0
ÿÿÿÿÿÿÿÿÿ3
predictions$
predictions:0	ÿÿÿÿÿÿÿÿÿ-
logits#
logits:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict