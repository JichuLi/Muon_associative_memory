"""Data-side interventions are constructed before and independently of learner keys."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import scipy.sparse as sp
import torch
from resources import atomic_json

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def entropy_columns(P):
    out = np.empty(P.shape[1], dtype=np.float64)
    for lo in range(0, P.shape[1], 128):
        a = P[:, lo:lo+128].astype(np.float64)
        out[lo:lo+128] = -np.sum(a * np.log(np.maximum(a, 1e-300)), axis=0)
    return out


def normalize(P, fallback=None):
    P = np.asarray(P, dtype=np.float32)
    sums = P.sum(axis=0, dtype=np.float64)
    empty = sums <= 0
    if empty.any():
        P[:, empty] = (np.full(P.shape[0], 1/P.shape[0]) if fallback is None else fallback)[:, None]
        sums = P.sum(axis=0, dtype=np.float64)
    P /= sums.astype(np.float32)[None, :]
    return P


def balance(P, pi, p, guard=None, tolerance=2e-8, maxiter=4000, warm=None):
    """Positive iterative scaling; preserve specified marginals, record residuals."""
    M = np.asarray(P, dtype=np.float64).copy()
    # Positive inputs must not acquire a new temperature-dependent probability floor.
    M = np.maximum(M, 1e-300 * np.maximum(p, 1e-30)[:, None])
    row_potential = np.zeros(len(p))
    if warm is not None and warm.get("row_potential") is not None:
        row_potential = warm["row_potential"].copy()
        row_potential -= row_potential.max()
        M *= np.exp(np.maximum(row_potential,-650))[:,None]
        M /= np.maximum(M.sum(0),1e-300)[None,:]
    M *= pi[None, :]
    positive = pi > 0
    error = math.inf
    for iteration in range(maxiter):
        ratio=p / np.maximum(M.sum(1), 1e-300)
        M *= ratio[:, None]
        row_potential += np.log(np.maximum(ratio,1e-300))
        M *= (pi / np.maximum(M.sum(0), 1e-300))[None, :]
        if iteration % 10 == 0:
            error = max(float(abs(M.sum(1)-p).max()), float(abs(M.sum(0)-pi).max()))
            tv_error=float(abs(M.sum(1)-p).sum()/2)
            if error < tolerance and tv_error<1e-7:
                break
        if guard is not None and iteration % 100 == 99:
            guard.checkpoint()
    if error >= tolerance or tv_error>=1e-7:
        raise RuntimeError(f"Marginal scaling did not converge: max error={error:.3g}, TV={tv_error:.3g}")
    if warm is not None:
        warm["row_potential"]=row_potential
    M[:, positive] /= pi[positive][None, :]
    M[:, ~positive] = p[:, None]
    return normalize(M.astype(np.float32), p), dict(balance_iterations=iteration+1,
        balance_max_abs_error=error,balance_total_variation=tv_error)


def entropy_match(P, pi, p, goal, guard=None):
    """Match entropy through a balanced exponential family, not clipping probabilities."""
    scores = np.log(np.maximum(P.astype(np.float64), 1e-10 * np.maximum(p,1e-30)[:,None]))
    scores -= scores.max(axis=0,keepdims=True)
    log_meta = {}
    warm={}
    def at(power):
        M = np.exp(np.clip(power * scores, -650, 0))
        result, info = balance(M, pi, p, guard,warm=warm)
        value = float(pi @ entropy_columns(result))
        return result, value, info
    low, high = 0.0, 1.0
    result, value, info = at(high)
    while value > goal and high < 128:
        high *= 2
        result, value, info = at(high)
    if value > goal + 2e-4:
        raise RuntimeError(f"Requested entropy {goal} not reached by balanced temperature family ({value}).")
    for _ in range(22):
        mid = (low+high)/2
        result, value, info = at(mid)
        if abs(value-goal) < 5e-5:
            break
        if value > goal:
            low = mid
        else:
            high = mid
    log_meta.update(info, entropy_goal=goal, entropy_achieved=value,
                    entropy_abs_error=abs(value-goal), entropy_temperature_power=mid)
    if log_meta["entropy_abs_error"] > 2e-4:
        raise RuntimeError(f"Entropy matching tolerance not satisfied: {log_meta}")
    return result, log_meta


@dataclass
class Target:
    P: np.ndarray
    pi: np.ndarray
    source_n: int
    indices: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def p(self):
        return self.P.astype(np.float64) @ self.pi

    def statistics(self):
        p = self.p
        ent = entropy_columns(self.P)
        ce = float(self.pi @ ent)
        return dict(n=self.P.shape[0], conditional_entropy=ce,
                    weighted_top1=float(self.pi @ self.P.max(0)),
                    target_entropy=float(-p @ np.log(np.maximum(p,1e-300))),
                    mutual_information=float(-p @ np.log(np.maximum(p,1e-300))-ce),
                    context_top1000_mass=float(np.sort(self.pi)[-min(1000,len(self.pi)):].sum()),
                    min_probability=float(self.P.min()),
                    max_column_sum_error=float(abs(self.P.sum(0,dtype=np.float64)-1).max()),
                    P_sha256=digest(self.P), pi_sha256=digest(self.pi),
                    **self.metadata)


class Corpus:
    def __init__(self, name="ptb", guard=None):
        self.name, self.guard = name, guard
        if name=="ptb":
            packaged = HERE/"data/ptb_v5000.pt"
            path = packaged if packaged.exists() else PROJECT/"stageA/data/ptb_v5000.pt"
        else:
            path = HERE/"data/wikitext2_v5000.pt"
        if not path.exists():
            raise FileNotFoundError(f"Required corpus not prepared: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = list(raw["vocab"])
        self.n = len(self.vocab)
        self.source_path = path
        self.source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        train = raw["splits"]["train"]
        y, x, counts = [train[k].numpy().astype(np.int64) for k in ["row_idx","col_idx","counts"]]
        self.counts = sp.coo_matrix((counts,(y,x)),shape=(self.n,self.n)).tocsc()
        self.total = int(counts.sum())
        self.pi = np.asarray(self.counts.sum(0)).ravel()/self.total
        self.p = np.asarray(self.counts.sum(1)).ravel()/self.total
        self.Q = (self.counts @ sp.diags(1/np.maximum(self.total*self.pi,1))).astype(np.float32).tocsc()
        self.splits = raw["splits"]
        self.asset_dir = HERE/"cache"/name
        self.asset_dir.mkdir(parents=True,exist_ok=True)

    def base(self, rho=.001):
        Q = self.Q.toarray()
        Q *= 1-rho
        Q += rho*self.p[:,None]
        return normalize(Q,self.p)

    def samples(self, seed):
        path = self.asset_dir/f"samples_{seed}.npz"
        if path.exists():
            return np.load(path)["labels"]
        rng = np.random.default_rng(20260901+seed)
        labels = np.empty((self.n,64),dtype=np.int32)
        for x in range(self.n):
            start,end = self.Q.indptr[x:x+2]
            rows,prob = self.Q.indices[start:end],self.Q.data[start:end].astype(np.float64)
            if prob.sum()>0:
                labels[x] = rng.choice(rows,size=64,p=prob/prob.sum())
            else:
                labels[x] = rng.choice(self.n,size=64,p=self.p)
        legacy = PROJECT/"research/ptb_onehot_quick_20260830/fixed_onehot_target.npz"
        if self.name=="ptb" and seed==0 and legacy.exists():
            labels[:,0] = np.load(legacy)["labels"]
        np.savez_compressed(path,labels=labels)
        return labels

    def sampled(self,k,seed):
        labels = self.samples(seed)[:,:k]
        Q = sp.coo_matrix((np.full(labels.size,1/k,dtype=np.float32),
                          (labels.ravel(),np.repeat(np.arange(self.n),k))),
                         shape=(self.n,self.n)).tocsc()
        return Q.toarray()

    def frequency_matched_labels(self, seed, rounds=1, bin_size=None):
        """Random context-label pairings whose pi-weighted output histogram tracks p."""
        rng = np.random.default_rng(93217+seed)
        labels = np.empty((self.n, rounds), dtype=np.int32)
        order = np.argsort(-0.5*(self.pi+self.p))
        width = int(bin_size or (2 if rounds == 1 else max(2, 2*rounds)))
        bins = [order[i:i+width] for i in range(0, self.n, width)]
        for r in range(rounds):
            current = np.empty(self.n, dtype=np.int32)
            for ids in bins:
                current[ids] = rng.permutation(ids)
            labels[:, r] = current
        return labels

    def frequency_blocks(self, k):
        """Contiguous output-frequency rank blocks with approximately equal p-mass."""
        order = np.argsort(-self.p)
        cumulative = np.cumsum(self.p[order])
        block = np.minimum((cumulative*k).astype(np.int32), k-1)
        labels = np.empty(self.n, dtype=np.int32)
        labels[order] = block
        return labels

    def rewired(self,seed,mask=None):
        rng = np.random.default_rng(20260829+seed)
        coo = self.counts.tocoo()
        xs = np.repeat(coo.col,coo.data)
        ys = np.repeat(coo.row,coo.data)
        if mask is None:
            ys = rng.permutation(ys)
        else:
            active = mask[xs]
            ys[active] = rng.permutation(ys[active])
        counts = sp.coo_matrix((np.ones(len(xs),dtype=np.float32),(ys,xs)),
                              shape=(self.n,self.n)).tocsc()
        assert np.array_equal(np.asarray(counts.sum(0)),np.asarray(self.counts.sum(0)))
        assert np.array_equal(np.asarray(counts.sum(1)),np.asarray(self.counts.sum(1)))
        return normalize((counts@sp.diags(1/np.maximum(self.total*self.pi,1))).toarray(),self.p)

    def groups(self,k,seed=0,random=False):
        path = self.asset_dir/f"groups_k{k}_random{int(random)}_seed{seed if random else 0}.npz"
        if path.exists():
            return np.load(path)["labels"]
        if random:
            labels = self.groups(k).copy()
            rng = np.random.default_rng(82711+seed)
            frequency_order = np.argsort(-self.pi)
            for ids in np.array_split(frequency_order,20):
                labels[ids] = rng.permutation(labels[ids])
        else:
            # Hellinger geometry, projected using data-only randomness.
            rng = np.random.default_rng(91412)
            Q = np.sqrt(self.base()).T
            projection = rng.normal(size=(self.n,48)).astype(np.float32)/np.sqrt(48)
            X = Q@projection
            del Q,projection
            centers = X[rng.choice(self.n,k,replace=False)].copy()
            labels = np.zeros(self.n,dtype=np.int32)
            for iteration in range(30):
                distance = np.sum(X*X,1)[:,None]+np.sum(centers*centers,1)[None,:]-2*X@centers.T
                new_labels = distance.argmin(1).astype(np.int32)
                if iteration and np.array_equal(new_labels,labels):
                    break
                labels = new_labels
                for c in range(k):
                    ids = labels==c
                    if ids.any():
                        centers[c] = np.average(X[ids],axis=0,weights=self.pi[ids])
                    else:
                        centers[c] = X[rng.integers(self.n)]
                if self.guard:
                    self.guard.checkpoint()
        np.savez_compressed(path,labels=labels)
        return labels

    def loglift(self,rank,random_middle,seed):
        path = self.asset_dir/"loglift_rank160.npz"
        if not path.exists():
            P = self.base()
            S = np.log(P.astype(np.float64)/self.p[:,None])
            S -= (self.p@S)[None,:]
            S -= (S@self.pi)[:,None]
            T = (np.sqrt(self.p)[:,None]*S*np.sqrt(self.pi)[None,:]).astype(np.float32)
            del P,S
            if self.guard:
                self.guard.checkpoint(force=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with torch.random.fork_rng(devices=[0] if device=="cuda" else []):
                torch.manual_seed(68237)
                A = torch.from_numpy(T).to(device)
                u,s,v = torch.pca_lowrank(A,q=min(160,self.n-1),center=False,niter=4)
                arrays = dict(u=u.cpu().numpy(),s=s.cpu().numpy(),v=v.cpu().numpy())
                del A,u,s,v
            np.savez_compressed(path,**arrays)
            del T
        with np.load(path) as f:
            u,s,v = f["u"][:,:rank].copy(),f["s"][:rank].copy(),f["v"][:,:rank].copy()
        if random_middle:
            rng = np.random.default_rng(88423+seed)
            lo = min(8,rank)
            # Keep the leading eight data modes; replace the remaining score directions.
            for a,mean in [(u,np.sqrt(self.p)),(v,np.sqrt(self.pi))]:
                z = rng.normal(size=(self.n,rank-lo))
                z -= mean[:,None]*(mean@z)[None,:]
                z -= a[:,:lo]@(a[:,:lo].T@z)
                a[:,lo:] = np.linalg.qr(z,mode="reduced")[0][:,:rank-lo]
        S = (u*s[None,:])@v.T
        S /= np.sqrt(self.p)[:,None]
        S /= np.sqrt(self.pi)[None,:]
        # Row/column potentials are irrelevant to subsequent balancing.
        S -= S.max()
        P = np.exp(np.clip(S,-80,0))
        return balance(P,self.pi,self.p,self.guard)[0]

    def text_counts(self,half,split_seed=86420):
        path = PROJECT/"stageA/data/raw/ptb.train.txt" if self.name=="ptb" else HERE/"data/wikitext2.train.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        order=np.random.default_rng(split_seed).permutation(len(lines))
        mask=np.zeros(len(lines),dtype=np.int8)
        mask[order[len(lines)//2:]]=1
        lines=[line for i,line in enumerate(lines) if mask[i]==half]
        mapping = {w:i for i,w in enumerate(self.vocab)}
        unk = mapping.get("<unk>",0)
        tokens = [mapping.get(w,unk) for line in lines for w in (line.strip().split()+["<eos>"])]
        ys,xs = np.asarray(tokens[1:]),np.asarray(tokens[:-1])
        return sp.coo_matrix((np.ones(len(xs)),(ys,xs)),shape=(self.n,self.n)).tocsc()

    def target(self,spec,seed=0):
        kind = spec["kind"]
        pi,p = self.pi.copy(),self.p.copy()
        indices = np.arange(self.n)
        meta = dict(corpus=self.name,source_sha256=self.source_hash,
                    target_recipe=spec,target_seed=seed)
        if kind=="ptb":
            P = self.base(spec.get("rho",.001))
        elif kind=="sample_k":
            P = self.sampled(spec["k"],seed)
        elif kind=="frequency_matched_onehot":
            labels = self.frequency_matched_labels(seed,1)[:,0]
            P = np.zeros((self.n,self.n),dtype=np.float32)
            P[labels,np.arange(self.n)] = 1
            meta["matched_label_sha256"] = digest(labels)
        elif kind=="frequency_matched_soft_k":
            k = spec["k"]
            labels = self.frequency_matched_labels(seed,k,spec.get("bin_size"))
            P = sp.coo_matrix((np.full(labels.size,1/k,dtype=np.float32),
                              (labels.ravel(),np.repeat(np.arange(self.n),k))),
                             shape=(self.n,self.n)).tocsc().toarray()
            meta["matched_label_sha256"] = digest(labels)
        elif kind=="argmax":
            y = np.asarray(self.Q.argmax(axis=0)).ravel()
            P = np.zeros((self.n,self.n),dtype=np.float32)
            P[y,np.arange(self.n)] = 1
        elif kind=="onehot_mix":
            P = (1-spec["alpha"])*self.sampled(1,seed)+spec["alpha"]*self.base(0)
        elif kind=="copy":
            a = spec["association"]
            P = np.broadcast_to((1-a)*pi[:,None],(self.n,self.n)).copy().astype(np.float32)
            P[np.arange(self.n),np.arange(self.n)] += a
        elif kind=="association":
            a = spec["alpha"]
            P = a*self.base(0)+(1-a)*p[:,None]
        elif kind=="rewire":
            a = spec["mix"]
            P = (1-a)*self.base()+a*(.999*self.rewired(seed)+.001*p[:,None])
        elif kind=="temperature":
            P = self.base().astype(np.float64)**(1/spec["tau"])
            P = normalize(P,p)
            if spec.get("balanced"):
                P,info = balance(P,pi,p,self.guard)
                meta.update(info)
        elif kind=="top_k":
            P = self.base(0)
            k = spec["k"]
            top = np.argpartition(P,-k,axis=0)[-k:]
            masked = np.zeros_like(P)
            cols = np.broadcast_to(np.arange(self.n)[None,:],top.shape)
            masked[top,cols] = P[top,cols]
            P = normalize(masked,p)
            background = spec["background"]
            P *= 1-background
            P += background*p[:,None]
        elif kind=="background_mix":
            Q = self.base()
            order = np.argsort(-p)
            if spec["bucket"]=="head":
                ids = order[:min(spec.get("count",1000),self.n)]
            elif spec["bucket"]=="tail":
                ids = order[-min(spec.get("count",2000),self.n):]
            else:
                raise ValueError(f"Unknown background bucket: {spec['bucket']}")
            background = np.zeros_like(p)
            background[ids] = p[ids]/p[ids].sum()
            a = spec["background"]
            P = (1-a)*Q+a*background[:,None]
            meta.update(background_bucket=spec["bucket"],background_ids=len(ids),
                        background_mass_in_source=float(p[ids].sum()))
        elif kind=="context_column_permutation":
            Q = self.base()
            rng = np.random.default_rng(76291+seed)
            order = rng.permutation(self.n)
            P = Q[:,order]
            meta["column_permutation_sha256"] = digest(order)
        elif kind=="column_permutation":
            P = self.base()
            rng = np.random.default_rng(72892+seed)
            for x in range(self.n):
                rng.shuffle(P[:,x])
            if spec.get("matched"):
                P,info = entropy_match(P,pi,p,float(pi@entropy_columns(self.base())),self.guard)
                meta.update(info)
        elif kind=="prototype":
            Q = self.base()
            groups = self.groups(spec["k"],seed,spec.get("random_groups",False))
            P = np.empty_like(Q)
            for group in np.unique(groups):
                ids = groups==group
                proto = Q[:,ids]@(pi[ids]/pi[ids].sum())
                P[:,ids] = proto[:,None]
            a = spec.get("residual",0)
            P = (1-a)*P+a*Q
            meta["group_labels_sha256"] = digest(groups)
        elif kind=="loglift":
            P = self.loglift(spec["rank"],spec.get("random_middle",False),seed)
        elif kind=="reverse":
            J = self.base()*pi[None,:]
            P = normalize(J.T.copy(),pi)
            pi = p.copy()
        elif kind=="symmetrize":
            P,info = balance(self.base(),pi,pi,self.guard)
            J = P*pi[None,:]
            a = spec["mix"]
            P = ((1-a)*J+a*(J+J.T)/2)/pi[None,:]
            meta.update(info,stationary_target_marginal="original context marginal")
            p = pi.copy()
        elif kind=="context_frequency":
            P = self.base()
            pi = pi**spec["gamma"]
            pi /= pi.sum()
        elif kind=="output_frequency":
            p = p**spec["gamma"]
            p /= p.sum()
            P,info = balance(self.base(),pi,p,self.guard)
            meta.update(info,retains="positive joint cross-product odds ratios")
        elif kind=="frequency_alignment":
            Q = self.base()
            ent = entropy_columns(Q)
            slots = np.argsort(-pi)
            ordering = np.argsort(-ent)
            if spec["alignment"]=="soft_low":
                ordering = ordering[::-1]
            elif spec["alignment"]=="random":
                ordering = np.random.default_rng(82877+seed).permutation(self.n)
            P = np.empty_like(Q)
            P[:,slots] = Q[:,ordering]
        elif kind=="entropy_context_alignment":
            Q = self.base()
            ent = entropy_columns(Q)
            slots = np.argsort(-pi)
            mode = spec["mode"]
            if mode=="high_frequency_low_entropy":
                ordering = np.argsort(ent)
            elif mode=="high_frequency_high_entropy":
                ordering = np.argsort(-ent)
            else:
                raise ValueError(f"Unknown entropy-context alignment: {mode}")
            P = np.empty_like(Q)
            P[:,slots] = Q[:,ordering]
            meta.update(entropy_alignment=mode)
        elif kind=="class_block_soft":
            blocks = self.frequency_blocks(spec.get("blocks",32))
            argmax = np.asarray(self.Q.argmax(axis=0)).ravel()
            background = spec.get("background",0.05)
            P = np.empty((self.n,self.n),dtype=np.float32)
            for block in np.unique(blocks):
                outputs = blocks==block
                dist = np.zeros(self.n,dtype=np.float32)
                dist[outputs] = p[outputs]/p[outputs].sum()
                contexts = blocks[argmax]==block
                P[:,contexts] = dist[:,None]
            P = (1-background)*P+background*p[:,None]
            meta.update(block_labels_sha256=digest(blocks),argmax_labels_sha256=digest(argmax),
                        blocks=int(spec.get("blocks",32)),background=background)
        elif kind in ["local_hardening","local_rewire"]:
            order = np.argsort(-pi)
            cut1,cut2 = min(1000,self.n//5),min(3000,3*self.n//5)
            group_ids = {"head":order[:cut1],"middle":order[cut1:cut2],"tail":order[cut2:]}[spec["group"]]
            mask = np.zeros(self.n,dtype=bool)
            mask[group_ids] = True
            mass = float(pi[mask].sum())
            strength = spec["mass_budget"]/mass
            if strength>1+1e-10:
                raise ValueError("Local intervention budget exceeds selected group's mass")
            P = self.base()
            replacement = self.sampled(1,seed) if kind=="local_hardening" else (.999*self.rewired(seed,mask)+.001*p[:,None])
            P[:,mask] = (1-strength)*P[:,mask]+strength*replacement[:,mask]
            meta.update(affected_context_mass=mass,intervention_strength=strength,
                        modified_probability_mass=mass*strength)
        elif kind in ["subsample","crossfit"]:
            if kind=="subsample":
                rng = np.random.default_rng(91682+seed)
                C = self.counts.copy()
                C.data = rng.binomial(C.data.astype(np.int64),spec["fraction"]).astype(np.float64)
                C.eliminate_zeros()
            else:
                C = self.text_counts(int(spec["split"][-1]),spec.get("split_seed",86420))
                meta["crossfit_split"]="Disjoint complete lines, randomized with a fixed data-only seed"
                meta["crossfit_split_seed"]=spec.get("split_seed",86420)
            total = float(C.sum())
            pi = np.asarray(C.sum(0)).ravel()/total
            p = np.asarray(C.sum(1)).ravel()/total
            P = normalize(C.toarray(),p)
            P = .999*P+.001*p[:,None]
            meta["effective_bigram_count"] = int(total)
        elif kind=="remove_tokens":
            remove = {i for i,w in enumerate(self.vocab) if w in spec["tokens"]}
            indices = np.asarray([i for i in range(self.n) if i not in remove])
            C = self.counts[indices][:,indices].copy()
            total = float(C.sum())
            pi = np.asarray(C.sum(0)).ravel()/total
            p = np.asarray(C.sum(1)).ravel()/total
            P = normalize(C.toarray(),p)
            P = .999*P+.001*p[:,None]
            meta.update(removed_ids=sorted(remove),effective_bigram_count=int(total))
        else:
            raise ValueError(f"Unknown target intervention: {kind}")
        if spec.get("match_entropy"):
            goal = float(self.pi@entropy_columns(self.base()))
            P,info = entropy_match(P,pi,p,goal,self.guard)
            meta.update(info)
        if "context_gamma" in spec:
            pi = pi**spec["context_gamma"]
            pi /= pi.sum()
        P = normalize(P,p)
        if P.min()<0 or not np.isfinite(P).all():
            raise ValueError("Illegal conditional distribution")
        target = Target(P=P,pi=pi,source_n=self.n,indices=indices,metadata=meta)
        if len(indices)==self.n:
            meta["target_marginal_tv_from_source"] = float(abs(target.p-self.p).sum()/2)
            meta["context_marginal_tv_from_source"] = float(abs(pi-self.pi).sum()/2)
        return target
