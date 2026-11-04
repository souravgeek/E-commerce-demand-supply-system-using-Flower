"""Flower ClientApp: each node is one store/warehouse that keeps its sales data local."""
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ecomfl.config import get_cfg
from ecomfl.task import DemandNet, evaluate_store, get_device, load_store, make_loader
from ecomfl.task import train as train_fn

app = ClientApp()


def _load(context: Context):
    partition_id = int(context.node_config["partition-id"])
    cfg = get_cfg(context.run_config)
    return load_store(partition_id, int(cfg["data-seed"]), float(cfg["heterogeneity"]))


@app.train()
def train(msg: Message, context: Context):
    model = DemandNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    data = _load(context)
    loader = make_loader(data.X_train, data.y_train, int(get_cfg(context.run_config)["batch-size"]), shuffle=True)

    cfg = msg.content["config"]
    train_loss = train_fn(
        model, loader,
        epochs=int(get_cfg(context.run_config)["local-epochs"]),
        lr=float(cfg["lr"]),
        device=device,
        proximal_mu=float(cfg.get("proximal-mu", 0.0)),  # injected by the FedProx strategy
    )

    content = RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": float(train_loss), "num-examples": len(loader.dataset)}),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = DemandNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    data = _load(context)
    m = evaluate_store(model, data, get_device())

    metrics = {
        "eval_loss": m["mse_log"], "eval_mae": m["mae"], "eval_wape": m["wape"],
        "fill_rate": m["fill_rate"], "stockout_rate": m["stockout_rate"],
        "avg_inventory": m["avg_inventory"], "total_cost": m["total_cost"],
        "num-examples": len(data.y_test),
    }
    content = RecordDict({"metrics": MetricRecord({k: float(v) if k != "num-examples" else int(v)
                                                    for k, v in metrics.items()})})
    return Message(content=content, reply_to=msg)
