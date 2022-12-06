import math
import copy
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def idNode(data, id_new_value_old):
    data = copy.deepcopy(data)
    data.x = None
    data.y[data.val_id] = -1
    data.y[data.test_id] = -1
    data.y = data.y[id_new_value_old]

    data.train_id = None
    data.test_id = None
    data.val_id = None

    id_old_value_new = torch.zeros(id_new_value_old.shape[0], dtype = torch.long)
    id_old_value_new[id_new_value_old] = torch.arange(0, id_new_value_old.shape[0], dtype = torch.long)
    row = data.edge_idxs[0]
    col = data.edge_idxs[1]
    row = id_old_value_new[row]
    col = id_old_value_new[col]
    data.edge_idxs = torch.stack([row, col], dim=0)

    return data

def shuffleData(data):
    data = copy.deepcopy(data)
    id_new_value_old = np.arange(data.shape[0])
    train_id_shuffle = copy.deepcopy(data.train_id)
    np.random.shuffle(train_id_shuffle)
    id_new_value_old[data.train_id] = train_id_shuffle
    data = idNode(data, id_new_value_old)

    return data, id_new_value_old


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    
    s_idx =  batch_size
    e_idx = min(num_instance, s_idx + batch_size)

    sources_batch = data.sources[s_idx: e_idx]
    destinations_batch = data.destinations[s_idx: e_idx]
    timestamps_batch = data.timestamps[s_idx:e_idx]
    edge_idxs_batch = edge_idxs[s_idx: e_idx]
    labels_batch = data.labels[s_idx: e_idx]
    source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
    
    
    
    node_id = np.arange(source_embedding.shape[0])
    np.random.shuffle(node_id)
    x = np.arange(100).reshape(2, 50)
    source_embedding.edge_idxs = torch.tensor( x, dtype= torch.int64)
    
    source_embedding.train_id = node_id[:int(source_embedding.shape[0] * 0.6)]
    source_embedding.val_id = node_id[int(source_embedding.shape[0] * 0.6):int(source_embedding.shape[0] * 0.8)]
    source_embedding.test_id = node_id[int(source_embedding.shape[0] * 0.8):]
    
    source_embedding.y = torch.tensor( labels_batch , dtype= torch.int64)
    data_b, id_new_value_old = shuffleData(source_embedding)
        
    lam = lam = np.random.beta(4.0, 4.0)
    labels_batch_torch = torch.from_numpy(labels_batch).float()
    pred_prob_batch = decoder(source_embedding, source_embedding.edge_idxs, data_b.edge_idxs, lam , id_new_value_old).sigmoid()
    pred_prob_batch = torch.flatten(pred_prob_batch)
    labels_batch_torch = torch.stack((1 - labels_batch_torch, labels_batch_torch), dim=-1)
    labels_batch_torch = torch.flatten(labels_batch_torch)  
    #pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()
    
  auc_roc = roc_auc_score(labels_batch_torch, pred_prob_batch)   
  return auc_roc