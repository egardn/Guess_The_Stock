Description of the project 

Challenge context

Founded in 1991, Capital Fund Management (CFM) is a successful alternative asset manager and a pioneer in the field of quantitative trading applied to capital markets across the globe. Our methodology relies on statistically robust analysis of terabytes of data to inform and enable our asset allocation, trading decisions and automated order execution. Our people’s diversity and dedication contribute to CFM’s unique culture of research, innovation and achievement. We are a Great Place to Work company and we offer a collaborative and informal work environment, attractive offices and facilities.

Challenge goals

The aim of this challenge is to attempt to identify from which stock a piece of tick-by-tick exchange data belongs. The problem is thus a classification task. The exchange data includes each atomic update of the order-book giving information about the best bid and offer prices, any trades that may have occurred, orders that have been placed in the book or cancelled. The order-book is also an aggregated order-book that is built out of multiple exchanges from where we can buy and sell the same shares. Although the data seems very non-descript and anonymous, we expect there to be clues in the data that will give away from which stock a piece of data belongs. This might be through the average spread, the typical quantities of shares at the bid or ask, the frequency with which trades occur, the distribution of how trades are split amongst the venues on which the stock is traded etc. there is a lot of information to aid the participant.

Data description

X: The dataset consists of 100 sequential order-book observations. There are 20 observations randomly taken per stock and per day. There are 504 days in the dataset (approximately 2 years) and 24 stocks. This means there are 100 x 20 x 505 x 24 rows of data = 24240000. The columns correspond to the following items:

obs_id: uniquely identifies a sequence of 100 order book events drawn from a random stock inside a random day;

venue: The exchange on which the event occurs. It can be NASDAQ, BATY, etc. but they are just encoded in the data as integers;

action: This is type of order-book event that occurred, it can be ‘A’, ‘D’ or ‘U’. A means volume was added to the book in the form of a new order. ‘D’ means an order was deleted from the book and ‘U’ means an order was updated;

order_id: The exchange data is ‘Level 3’or Market-by-Order, this means that each update provides a unique identifier for the specific order that was affected. It means that we can track the lifetime of an individual order. If it was placed earlier with a ‘A’, we may see it again deleted in the data by the same market participant if we see the same order id occur again with a ‘D’. Note however that the order-ids have been obfuscated somewhat. The first order referenced in any given sequence of data for a particular observation is given the id=0. If order_id 0 is seen again, you will know that it was the same order again that was affected;

side: The side of the order-book on which the event took place ‘A’ or ‘B’;

price: The price of the order that was affected;

bid: The price of the best bid;

ask: The price of the best ask;

bid_size: The volume of orders at the best bid of the aggregated book;

ask_size: The volume of orders at the best ask of the aggregated book;

flux: The change to the order-book affected by the event. i.e. if the volume in a level increased or decreased due to the event;

trade: A boolean true or false to indicate whether a deletion or update event was due to a trade or due to a cancellation.

Because the price itself provides such a large clue, we subtract the best bid price for the first event in the sequence of 100 from the ‘price’, ‘bid’ and ‘ask’ fields.

Y: The Y of the dataset is the eqt_code_cat. However, for the training set construction this is an integer between 0 and 23 which identifies the particular stock that was affected.

The training set is drawn from one period of time. The same stocks are used again in the test period, but the observations of the market are drawn from a different future period.

Benchmark description

Model architecture and feature construction

First the inputs are pre-processed to produce a tensor of shape (100, 30) for each observation (100 events, and 30 dimensional input vector).

The 30 dimensional input vector is made out:

Embedding of venue [8];

Embedding of action [8];

Embedding of trade [8];

Bid [1];

Ask [1];

Price [1];

log(bid_size + 1) [1];

log(ask_size + 1) [1];

log(flux) [1].

This tensor is processed with two 64-dimension GRU cells:

One which passes forwards through the data, producing a 64-dimensional output;

One which passes backwards through the data, producing a 64-dimensional output.

The final output of the two GRU sequence models is concatenated into a 128-dimensional vector.

This 128 dimensional vector is then further processed with two dense layers:

One condenses the 128 dimensions to 64 with a linear model and applies a SeLU activation function;

The Linear combines the 64 outputs after the selu activation into 24 final logit values from which we can apply a softmax layer to produce category probabilities.

Training

The loss is the cross-entropy as is standard for classification tasks. 1000 randomly chosen observations are drawn from the training set to build batches of shape (1000, 100, 30) on which we calculate the loss gradient. The model is trained by stochastic gradient with the Adam optimizer (default Optax parameters) with a learning rate of 3e-3. The training continues for 10000 batches.