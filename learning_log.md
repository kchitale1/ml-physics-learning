Day A:
Learning python class definition and defining methods. How self is used internally to the class. __main__ block guard for when num_workers>0 for the future. Return the output of operations as as instance of the class itself is allowed. Capital letters for class names is convention. 

Day B:
Dunder methods also called as special methods in python. __len__ and __getitem__ important for future on Dataset. __repr__ should be used default instead of __str__ by developers. __getitem__ gives free iteration if implemented correctly. 

Day C:
Inheritance. Defining subclass from a class. Can override original methods. super() used inside child's methods to use parent's method instead of replacing. In Dataset class __len__ and __getitem__ are not defined and have to be defined in the subclass. This is standard in pytorch that main class provides framework and main method has to be defined by the subclass. self is subclass instance even inside the parent methods so polymorphism works. Reflection or introspection: "let me decide what I am at runtime". Method resolution order by doing __mro__. Dataset provides the contract, DataLoader provides the machinery

Day D:
lists, iterables: new_list = [expression for member in iterable] 
To add conditionals: 
new_list = [expression for member in iterable if conditional] or
new_list = [true_expr if conditional else false_expr for member in iterable]
Nesting is possible eg. [n for n i num for let in letters] etc
zip creates tuples that match indices of passed lists. Lists with square brackets, dictionaries with curly brackets. Set has unique values only with curly brackets. Generator with curly brackets. Dictionaries have key:value system than indenxing for lists. Use zip_longest when there is index mismatch. starred unpacking takes everything that is remaining from iterable. Use str.format() for reusable template. f-strings can do arithmatic but need variables present at runtime when the line is executed. Use %s style for logging

Day E:
Generators use yield to pause to save memory. It remembers the state of the function. Can be used in comprehension in curly brackets, similar form in list will use entire memory by building the full list. Use next() to step. A function with yield anywhere in its body becomes a generator. Usually called with a look but yield will make sure the function only progresses when user asks. "with" is for context manager, opening files etc to handle exceptions (analog to try: and finally:) 
with <expression> as <name>:
    <body>  
Calling a generator function does not really execute it but just constructs the object, it only rusn with next(gen_fun) is called upto yield. sum(), list() etc run to exhaustion regardless of number of yields. Thus be careful when having while True in a loop and calling sum() on it. 
islice is iterator slice: islice(iterable, start, stop, step). It returns an iterator and not a list (lazy). Need to be wrapped in list if entire thing is needed. Used in streaming data to get a defined block eg list(islice(generator, 100))

Day F:
Dataset and fake loader
A training-dataset's __getitem__(i) returns everything the loss needs to evaluate one example — input and the correct answer.
for x, y in loader:
A FakeLoader wraps a dataset, and when you iterate it, you get one batch at a time. Each batch is a chunk of batch_size examples, optionally in shuffled order.
DataLoader init gets batch size and shuffle true or fales. iter function iterates over the batch size and builds the lists.
Refer Dataset_learning.py

Day G:
Annotated MINST code. Importance of calling super(), dunders used. How Dataset and DataLoader interact etc. What context manager with no_grad does etc. 

Day 4:
Back to physics AI learning. Building 1D diffusion problem. Defining Dataset class and using pytorch DataLoader. If num_workers>0 for windows, the code needs to be in if __name__=='__main__' guard as forking is not available. 
Dataset has one job: given an index, return one sample. It knows nothing about batches, shuffling, or parallelism. Your DataLoader has one job: take a Dataset and turn it into a stream of batches. It knows nothing about what the data actually is.
zip(*batch) transposes a list of (input, target) pairs into two tuples — one of all inputs, one of all targets. The * is the key: it converts the list into positional arguments so zip can do its column-grouping. ie you get one tensor for all inputs and one for all outputs after stacking. 
Custom collate function can also pad for different sizes. 

Day 5:
Losses. MSE, MAE and L2 loss. torch.norm is by default L2 norm. softmax+NLL in the same op gives numerical stability. state_dict is your data; your code is your architecture. Keep them separate.
model.train() turns Dropout to True which randomly shuts off some of the neurons during train step. This is to regularize the model so it does not rely on a specific pathway all the time. During evaluation you want ALL NEURONS ON and model.eval() tells it to not do Dropout. 2nd is BatchNorm. For training mean and std deviation is derived from the whole batch so works fine but during evaluation we are doing sample by sample, so during eval it uses frozen in time long term mean from training. 
state_dict is used to save snapshot of the model parameters. eg saving the best model from iterations. torch.save to save the snapshop and reload later. 

Day 6:
torch.device. Check what device is available then Use .to(device) to send model and data batch to a device (usually gpu) in training loop. For total loss use loss.item() instead of just loss to detach from the graph and get a pure python float. Otherwise on GPU with many batches the loss will accumulate. model.eval() turns off dropout and fixes BatchNorm, and torch.no_grad() stops gradient tracking
On GPUs to get speedup use autocast to use dtype of torch.bfloat16. Half precision. Better than float16 due to wider exponent range so it has the same range as float32. Use bfloat16 only on the forward pass on matmul etc expensive ops. Backward and optimizer needs to run in full precision. For profiling use cuda events. eval doesn't need autocast as no gradients are tracked but stil needs .to(device) for batches.  

Day 7:
Coding full sin*exp analytical function NN with different number of layers and parameter count. Underfit obvious with tiny model but large (with #parameters>>inputs) still doesn't overfit because of early stopping and implicit regularization (dam's optimizer trajectory bias toward flat minima, the geometry of overparameterized loss landscapes, and the smoothness of MSE on a smooth target). Always look at GAP between train_loss and val_loss to determine overfitting. Since we used Gaussian noise, there is a noise floor that the model can't beat. 
Overparameterized networks don't reliably overfit in practice — early stopping and implicit regularization bias the optimizer toward generalizing solutions, often violating classical "params >> data → memorize" intuition. To diagnose what's happening, look at three things together: the train-val gap (overfit signature), val loss anchored to the noise floor σ²_noise (capacity adequacy), and σ of residuals (independent capacity check). Val curve alone isn't enough.
Width of MLP is #of neurons in a layer and depth is number of hidden layers. MLP's parameter count scales roughly as depth × width²
Total expected error = Bias² + Variance + σ²_noise
Always predict noise floor to see how well the model is doing. 
Noise floor matters to val_loss. There is no therotical floor for train_loss and model can find parameters to represent each sample point perfectly. 
Regularize when overfit, scale up when underfit
σ²(resid) = σ²(missed) + σ²(noise)

Day 8:
Weight initialization: so that weights don't explore or become zero as you go deeper into the network. Batchnorm re-centers and rescales the output distributions after each layer on forward pass. Var(y_i) = N · Var(w) · Var(x). Here N.Var(w) compounds over every pass. If we call this alpha the multiplication accumulates over each pass so if alpha > 1, variance increases every layer. If alpha < 1, variance goes to zero very fast. Optimally you want alpha near 1. This is what batchnorm is doing. This variance is important during backward pass resulting in exploding or vanishing gradients and weights barely moving. 
Main takeaway:  without stable activations, you can't get a meaningful gradient signal to the early layers, and without that, deep networks don't learn any differently from shallow ones. The whole point of depth is that each layer can learn increasingly abstract features; that only happens if every layer actually receives a useful gradient.
BN makes sure activations are normalized but it can't fix bad initialization. With good initialization BN has lower impact. BN shines more in DEEPER networks where variance compounding is a larger problem. ReLU zeros half of the neurons in forward pass so He account for that by introducing factor of 2. This is due to ReLU giving out 0 for negative ouputs so on average half neurons would be off (separate from dropout). Bsad init lands in a basin with a higher floor. ayerNorm when you have sequences or variable batch sizes. BatchNorm when you have fixed-length vectors and normal batch training.
With BN, problem will converge faster but with bad initilization will have a higher loss floor. No BN with good initialization will converge fine but takes longer. 

Day 9: CNN
Channels, pooling, convolution (filter), receptive field
 conv layers learn what to look for (filters), and get deeper with more channels as you go. Pool layers shrink the spatial dimensions so later layers see larger regions of the original image without the cost of giant kernels.
 Dropout to reduce overfitting but trains slower. Expected accuracy is around 70% with validation set with CIFAR-10 dataset. 
 RF = RF_prev + (kernel_size − 1) × stride_product
Stride of 1 is the standard choice


Day 10: wandb
Hyperparameters are important. One bad hyperparameter can ruin regardless of choice of others so select learning rate and batch size smartly. 
Smaller batch means noisy gradient combined with a large step size results in optimizer overshooting,. gradient noise and step size interact multiplicatively. 
Gradient graphs on wandb are useful to see vanishing

Day 11: CFD problem
Created a csv for velocity, pressure, vorticity dataset with openfoam and trained MLP to predict vorticity. Needed to include multiple timesteps to improve the fit of prediction vs actual vorticity. MLP struggles with extreme values eg vorticity near walls so Rsq log space is more important. Random splits of training and validation across all times are dangerous because adjacent time steps are correlated so model learns this and gets rewarded artificially increasing Rsq. Proper split would be temporal, ie train on early timesteps and validate on the latter eg. 
Physically vorticity depends on gradients of velocity and not velocity itself so there is a ceiling in how much we can learn from single point training. A graph network or CNN can learn spatial features and would be able to break through the ceiling.
Rsq of negative means worse prediction than just predicting the mean. 

Day H: Gaussian Processes
A GP is a distribution over functions. When you sample from a GP, you don't get a number — you get a whole function (a curve)
Mean function: starting point estimate, usually0
Covariance function (kernel): how similar are the function values at two input points x and x'?
The most common kernel is the RBF (radial basis function), also called the squared exponential:
k(x, x') = exp(−|x − x'|² / 2ℓ²)
if x and x' are close together, k ≈ 1 (outputs should be similar). If they're far apart, k ≈ 0 (outputs can be anything). The length scale ℓ controls what "close" means
If A and B have high positive covariance, knowing A is large tells you B is probably large too. They move together.
If covariance is near zero, knowing A tells you nothing about B.
Negative covariance means they move opposite
So k(x, x') answers: "If I know the function value at x, how much does that constrain what the function can be at x'?"
High k → knowing f(x) tells me a lot about f(x'). The function can't jump wildly between x and x'.
Low k → f(x) and f(x') are nearly independent. The function could be anything at x' even if I know x.
:::::::::::Gaussian Process — Mental Model:::::::::::

Start with a mean function m(x), typically set to zero. This is your prior belief about the function before seeing any data — not a committed fit, just a neutral starting point.

Choose a kernel k(x, x') that encodes your prior belief about the function's shape — specifically, how correlated nearby function values should be. The RBF kernel says the function is smooth: values at nearby inputs move together, values far apart are nearly independent. The length scale ℓ controls how quickly that correlation decays.
Given training data at points X with outputs y, compute the covariance matrix K where each entry K[i,j] = k(xᵢ, xⱼ). This matrix captures the pairwise correlations between all your observations.

To predict at a new point x*, condition the GP on the training data. This gives you an exact analytical expression for the posterior mean (your best guess at x*) and posterior variance (your uncertainty). No sampling required. The posterior mean is a weighted average of training outputs, where the weights reflect how similar x* is to each training point. The posterior variance is small near training data and large far from it.

If observation noise σ² is included, the GP doesn't pass exactly through training points — it interpolates smoothly with residual uncertainty even at observed locations, reflecting measurement error. If noise is zero, the posterior variance at any training point collapses to exactly zero.

Sampling from the GP (optional) draws complete plausible functions consistent with the data — useful for visualization and understanding uncertainty, but not needed for prediction itself.
:::::::::::::::::::::::::::::::::::::::::::::::::::
length_scale ℓ controls how quickly correlation decays with distance. Small ℓ means even nearby points are treated as nearly independent — the function can wiggle rapidly. Large ℓ means points far apart are still correlated — the function is forced to be slowly varying and smooth. It's essentially your prior belief about the spatial scale of variation in your function.
noise_var σ² controls how much you trust your observations. High σ² means "my measurements have significant error, don't fit through them exactly." Low σ² means "my data is clean, fit close to every point." At σ²=0 you get exact interpolation.
noise_car can be an array so that each observation has a specific variance from experiments.    
The prior over functions is Gaussian. The observation noise is Gaussian. Gaussian prior × Gaussian likelihood = Gaussian posterior — and that product has a closed-form expression. No approximation needed. If you used a non-Gaussian kernel or non-Gaussian noise, the posterior would no longer be Gaussian and you'd need approximations like variational inference or MCMC. 
Isotropic vs ARD kernel for multi dimensional inputs. ARD preferred as uses different lengths. 
Normalize the input data with variance of 1 so that GP doesn't get confused and give one input more weight than the other.

Day H: GPytorch
Kernel, conditioning and conjugate. GPs are special because Gaussian is a special function. 
Algorithm: (l is assumed 1 below)
1. Calculate covariance matrix between the training data (K + σ_n²I)
2. For new point calculate covariance vector with training points: K*
3. Calculate prior variance K** (usually 1 since new point's covariance with itself is 0)
4. Solve for α = (K + σ_n²I)⁻¹ y. This is using training data. This operation costs O(n^3) in compute
5. The mean prediction at new point is μ∗​=K*^T​α and variance at new point is σ∗2​=K**​−K*^T(K + σ_n²I)⁻¹K*
The GP prior exists before seeing any data. It represents your beliefs about what functions are plausible given only your assumptions about smoothness, scale, etc. (encoded by the kernel). The data comes in later when you compute the posterior.
Small l means more wiggly functions, large l means smoother. 




