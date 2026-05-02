import numpy as np
from sklearn.preprocessing import StandardScaler
from dynamic_features import stack_context

split = np.load("splitdata.npz", allow_pickle=True)
train_items = split["train_items"]
val_items = split["val_items"]
stateList = split["stateList"]

testfile = np.load("testdata.npz", allow_pickle=True)
test_items = testfile["data"]


def flatten_items(items, feature_type="lmfcc", dynamic=False):
    xs = []
    ys = []

    for item in items:
        x = item[feature_type]

        if dynamic:
            x = stack_context(x)

        y = item["targets"]

        xs.append(x)
        ys.append(y)

    x_all = np.vstack(xs)
    y_all = np.concatenate(ys)

    return x_all, y_all


# Static MFCC
lmfcc_train_x, train_y = flatten_items(train_items, "lmfcc", dynamic=False)
lmfcc_val_x, val_y = flatten_items(val_items, "lmfcc", dynamic=False)
lmfcc_test_x, test_y = flatten_items(test_items, "lmfcc", dynamic=False)

# Static mspec
mspec_train_x, _ = flatten_items(train_items, "mspec", dynamic=False)
mspec_val_x, _ = flatten_items(val_items, "mspec", dynamic=False)
mspec_test_x, _ = flatten_items(test_items, "mspec", dynamic=False)

# Dynamic MFCC
dlmfcc_train_x, _ = flatten_items(train_items, "lmfcc", dynamic=True)
dlmfcc_val_x, _ = flatten_items(val_items, "lmfcc", dynamic=True)
dlmfcc_test_x, _ = flatten_items(test_items, "lmfcc", dynamic=True)

# Dynamic mspec
dmspec_train_x, _ = flatten_items(train_items, "mspec", dynamic=True)
dmspec_val_x, _ = flatten_items(val_items, "mspec", dynamic=True)
dmspec_test_x, _ = flatten_items(test_items, "mspec", dynamic=True)


def standardize(train_x, val_x, test_x):
    scaler = StandardScaler()

    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    return (
        train_x.astype("float32"),
        val_x.astype("float32"),
        test_x.astype("float32"),
        scaler
    )


lmfcc_train_x, lmfcc_val_x, lmfcc_test_x, lmfcc_scaler = standardize(
    lmfcc_train_x, lmfcc_val_x, lmfcc_test_x
)

mspec_train_x, mspec_val_x, mspec_test_x, mspec_scaler = standardize(
    mspec_train_x, mspec_val_x, mspec_test_x
)

dlmfcc_train_x, dlmfcc_val_x, dlmfcc_test_x, dlmfcc_scaler = standardize(
    dlmfcc_train_x, dlmfcc_val_x, dlmfcc_test_x
)

dmspec_train_x, dmspec_val_x, dmspec_test_x, dmspec_scaler = standardize(
    dmspec_train_x, dmspec_val_x, dmspec_test_x
)

train_y = train_y.astype("int64")
val_y = val_y.astype("int64")
test_y = test_y.astype("int64")

np.savez(
    "prepared_data.npz",

    lmfcc_train_x=lmfcc_train_x,
    lmfcc_val_x=lmfcc_val_x,
    lmfcc_test_x=lmfcc_test_x,

    mspec_train_x=mspec_train_x,
    mspec_val_x=mspec_val_x,
    mspec_test_x=mspec_test_x,

    dlmfcc_train_x=dlmfcc_train_x,
    dlmfcc_val_x=dlmfcc_val_x,
    dlmfcc_test_x=dlmfcc_test_x,

    dmspec_train_x=dmspec_train_x,
    dmspec_val_x=dmspec_val_x,
    dmspec_test_x=dmspec_test_x,

    train_y=train_y,
    val_y=val_y,
    test_y=test_y,

    stateList=stateList
)

print("saved prepared_data.npz")
print("lmfcc_train_x:", lmfcc_train_x.shape)
print("mspec_train_x:", mspec_train_x.shape)
print("dlmfcc_train_x:", dlmfcc_train_x.shape)
print("dmspec_train_x:", dmspec_train_x.shape)
print("train_y:", train_y.shape)