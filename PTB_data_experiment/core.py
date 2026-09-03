"""Full-batch CE oracle, numerical rays and frozen-geometry controls."""
from __future__ import annotations
import math
import numpy as np
import torch
from targets import digest


def representations(n,d,seed,device,dtype=torch.float32):
    generator = torch.Generator(device=device).manual_seed(seed)
    E = torch.randn((d,n),generator=generator,device=device,dtype=dtype)
    U = torch.randn((n,d),generator=generator,device=device,dtype=dtype)
    E /= torch.linalg.vector_norm(E,dim=0,keepdim=True)
    U /= torch.linalg.vector_norm(U,dim=1,keepdim=True)
    return E,U


def symmetric_power(A,power):
    values,vectors = torch.linalg.eigh(A.double())
    if float(values.min())<=0:
        raise ValueError("Initial covariance is not positive definite; no hidden ridge is applied.")
    return (vectors*values.pow(power)[None,:])@vectors.T


def polar(G,rtol):
    if float(torch.linalg.vector_norm(G))==0:
        return torch.zeros_like(G),torch.zeros(G.shape[0],device=G.device,dtype=G.dtype)
    u,s,vh = torch.linalg.svd(G,full_matrices=False)
    keep = s>rtol*s[0]
    return u[:,keep]@vh[keep,:],s


class Problem:
    def __init__(self,target,case,rep_seed,config,device="cuda",dtype=torch.float32):
        self.target,self.case,self.config = target,case,config
        self.device,self.dtype = torch.device(device),dtype
        self.d = config["d"]
        E,U = representations(target.source_n,self.d,rep_seed,self.device,dtype)
        ids = torch.as_tensor(target.indices,device=self.device)
        self.E,self.U = E[:,ids].contiguous(),U[ids].contiguous()
        self.feature_hashes = dict(E=digest(self.E.cpu().numpy()),U=digest(self.U.cpu().numpy()))
        self.P = torch.as_tensor(target.P,device=self.device,dtype=dtype)
        self.pi = torch.as_tensor(target.pi,device=self.device,dtype=dtype)
        self.pi64 = self.pi.double()
        self.pi64 /= self.pi64.sum()
        self.pi = self.pi64.to(dtype)
        self.column_mass = self.P.sum(0,dtype=torch.float64).to(dtype)
        self.p = (self.P.double()@self.pi64).to(dtype)
        self.p /= self.p.sum()
        self.initial_bias = self.make_bias(case["bias"],target.metadata["target_seed"])
        self.bias = self.initial_bias.clone()
        self.learnable_bias = case["bias"]["kind"]=="learnable"
        self.permutation = self.inverse_permutation = None
        self.refresh_initial_geometry()
        geometry = case["geometry"]
        rng = np.random.default_rng(271828+target.metadata["target_seed"])
        if geometry["kind"]=="whiten":
            side = geometry["side"]
            if side in ["input","both"]:
                self.E = (symmetric_power(self.C,-.5)@self.E.double()).to(dtype)
            if side in ["output","both"]:
                self.U = (self.U.double()@symmetric_power(self.A,-.5)).to(dtype)
            self.refresh_initial_geometry()
        elif geometry["kind"]=="orthogonal":
            a,b = [np.linalg.qr(rng.normal(size=(self.d,self.d)))[0] for _ in range(2)]
            self.U = self.U@torch.as_tensor(a,device=self.device,dtype=dtype)
            self.E = torch.as_tensor(b,device=self.device,dtype=dtype)@self.E
            self.refresh_initial_geometry()
        elif geometry["kind"]=="entry_permutation":
            permutation = np.arange(self.d**2)
            subset = rng.permutation(self.d**2)[:int(round(geometry["fraction"]*self.d**2))]
            permutation[subset] = rng.permutation(subset)
            self.permutation = torch.as_tensor(permutation,device=self.device)
            self.inverse_permutation = torch.as_tensor(np.argsort(permutation),device=self.device)
        elif geometry["kind"]!="original":
            raise ValueError(geometry)
        self.Ainv,self.Cinv = symmetric_power(self.A,-1),symmetric_power(self.C,-1)
        self.B = self.U.T@((self.P*self.pi[None,:])@self.E.T)
        self.initial_cond = float(torch.linalg.cond(self.A)*torch.linalg.cond(self.C))
        self.mean_E = self.E@self.pi
        order = np.argsort(-target.pi)
        boundaries = [0,len(order)//5,3*len(order)//5,len(order)]
        self.buckets = [torch.as_tensor(order[a:b],device=self.device) for a,b in zip(boundaries[:-1],boundaries[1:])]
        self.bucket_mass = [float(self.pi[ids].sum()) for ids in self.buckets]
        self.heldout = None

    def attach_evaluation_counts(self,rows,columns,counts,label):
        values = torch.as_tensor(np.asarray(counts),device=self.device,dtype=torch.float64)
        self.heldout = (torch.as_tensor(np.asarray(rows),device=self.device),
                        torch.as_tensor(np.asarray(columns),device=self.device),values/values.sum(),label)

    def make_bias(self,spec,seed):
        p = self.p.double().clamp_min(1e-30)
        kind = spec["kind"]
        if kind=="zero" or (kind=="learnable" and spec["initial"]=="zero"):
            b = torch.zeros_like(p)
        elif kind=="power":
            b = spec["beta"]*p.log()
        elif kind=="learnable":
            b = p.log()
        elif kind in ["shuffled","within_bins"]:
            values = p.cpu().numpy()
            rng = np.random.default_rng(64922+seed)
            if kind=="shuffled":
                values = values[rng.permutation(len(values))]
            else:
                values = values.copy()
                for ids in np.array_split(np.argsort(-values),10):
                    values[ids] = rng.permutation(values[ids])
            b = torch.as_tensor(np.log(values),device=self.device)
        elif kind=="groups":
            values = p.cpu().numpy()
            q = np.empty_like(values)
            for ids in np.array_split(np.argsort(-values),spec["groups"]):
                q[ids] = values[ids].sum()/len(ids)
            b = torch.as_tensor(np.log(q),device=self.device)
        else:
            raise ValueError(f"Unknown bias: {spec}")
        b -= b.mean()  # A common output potential does not change any probabilities.
        return b.to(self.dtype)

    def refresh_initial_geometry(self):
        q = self.initial_bias.double().softmax(0)
        U,E = self.U.double(),self.E.double()
        centered = U-(q@U)[None,:]
        self.A = centered.T@(q[:,None]*centered)
        self.C = (E*self.pi64[None,:])@E.T

    def reset(self):
        self.bias = self.initial_bias.clone()
        return torch.zeros((self.d,self.d),device=self.device,dtype=self.dtype)

    def coordinates(self,G):
        return G if self.permutation is None else G.flatten()[self.inverse_permutation].reshape_as(G)

    def physical(self,D):
        return D if self.permutation is None else D.flatten()[self.permutation].reshape_as(D)

    @torch.no_grad()
    def evaluate(self,W):
        z = self.U@(W@self.E)+self.bias[:,None]
        # Center by a maximum, so CE is a sum of nonnegative terms even at large steps.
        z = z-z.amax(0,keepdim=True)
        logz = torch.logsumexp(z,dim=0)
        q = z.softmax(0)
        expected = (self.P*z).sum(0,dtype=torch.float64)
        ce = logz.double()*self.column_mass.double()-expected
        loss = float(self.pi64@ce)
        residual = (q*self.column_mass[None,:]-self.P)*self.pi[None,:]
        G = self.U.T@residual@self.E.T
        gb = residual.sum(1)
        spec = self.case["target"]
        if (spec["kind"]=="prototype" and spec.get("residual",0)==0
                and float(torch.linalg.vector_norm(W))==0):
            # Repeated conditional columns imply rank(G0) <= number of groups.
            # Accumulate before casting: float32 GEMM errors otherwise cross
            # the polar cutoff and create artificial null-space directions.
            residual64=(q.double()*self.column_mass.double()[None,:]-self.P.double())*self.pi.double()[None,:]
            # Keep this small matrix in double through the polar SVD as well:
            # casting before a float32 SVD can still invent threshold-scale modes.
            G=self.U.double().T@residual64@self.E.double().T
        if spec["kind"]=="association" and spec.get("alpha")==0 and float(torch.linalg.vector_norm(W))==0:
            # The independent target has a rank-one gradient at zero.
            # Form the outer product directly so float32 GEMM noise is not
            # promoted to spurious polar directions around the 1e-7 cutoff.
            delta=q[:,0]*self.column_mass[0]-self.P[:,0]
            G=torch.outer(self.U.T@delta,self.mean_E)
        if (spec["kind"]=="association" and spec.get("alpha")==0
                and self.case["bias"].get("kind")=="power" and self.case["bias"].get("beta")==1
                and float(torch.linalg.vector_norm(W))==0):
            G.zero_()
            gb.zero_()
        return dict(loss=loss,G=G,gb=gb,z=z,q=q,ce=ce)

    def direction(self,G,method):
        base = method.split("_")[0]
        singular = None
        if base=="gd":
            D = -G
        elif base=="ngd":
            D = -G/torch.linalg.vector_norm(G).clamp_min(1e-30)
        elif base=="muon":
            D,singular = polar(self.coordinates(G),self.config["polar_rtol"])
            D = -self.physical(D)
        elif base=="h0":
            D = -(self.Ainv@G.double()@self.Cinv).to(self.dtype)
        elif base=="signgd":
            D = -torch.sign(G)
        else:
            raise ValueError(method)
        return D.to(self.dtype),singular

    def preconditioned(self,G,kind):
        g = G.double()
        if kind=="input":
            return -(g@self.Cinv).to(self.dtype)
        if kind=="output":
            return -(self.Ainv@g).to(self.dtype)
        if kind=="diagonal":
            return -(g/(self.A.diag()[:,None]*self.C.diag()[None,:])).to(self.dtype)
        if kind=="wrong_basis":
            generator = torch.Generator(device=self.device).manual_seed(92348)
            Q = torch.linalg.qr(torch.randn(g.shape,generator=generator,device=self.device,dtype=torch.float64))[0]
            return -(self.Ainv@g@Q@self.Cinv@Q.T).to(self.dtype)
        if kind in ["reference_zero","reference_unigram"]:
            q = torch.ones_like(self.p).double()/len(self.p) if kind=="reference_zero" else self.p.double()
            U = self.U.double()
            centered = U-(q@U)[None,:]
            A = centered.T@(q[:,None]*centered)
            return -(symmetric_power(A,-1)@g@self.Cinv).to(self.dtype)
        raise ValueError(kind)

    @torch.no_grad()
    def direction_geometry(self,evaluation,D):
        norm = float(torch.linalg.vector_norm(D))
        if norm==0:
            return dict(direction_norm=0.,slope=0.,curvature=0.,initial_curvature=0.),None
        unit = D/norm
        S = self.U@(unit@self.E)
        q = evaluation["q"]
        mean = (q*S).sum(0)
        variance = (q*(S-mean[None,:]).square()).sum(0)
        curvature = float(self.pi64@variance.double())
        slope = -float((evaluation["G"].double()*unit.double()).sum())
        unit64 = unit.double()
        initial_curvature = float((unit64*(self.A@unit64@self.C)).sum())
        return dict(direction_norm=norm,slope=slope,curvature=curvature,initial_curvature=initial_curvature),S

    @torch.no_grad()
    def ray(self,W,evaluation,D,geometry=None):
        info,S = self.direction_geometry(evaluation,D) if geometry is None else geometry
        norm = info["direction_norm"]
        start = evaluation["loss"]
        if norm==0 or info["slope"]<=0:
            return W.clone(),dict(**info,eta=0.,eta_fro=0.,line_evals=0,
                bracketed=norm==0,terminal_derivative=-info["slope"],
                ray_loss=start,relative_bracket_width=0.,ray_gain=0.)
        z0 = evaluation["z"]
        target_base = float(self.pi64@(self.P*z0).sum(0,dtype=torch.float64))
        target_slope = float(self.pi64@(self.P*S).sum(0,dtype=torch.float64))
        def at(alpha,need_loss=True):
            z = z0+alpha*S
            q = z.softmax(0)
            loss = (float(self.pi64@(torch.logsumexp(z,0).double()*self.column_mass.double()))-target_base-alpha*target_slope
                    if need_loss else math.nan)
            derivative = float(self.pi64@((q*S).sum(0).double()*self.column_mass.double()))-target_slope
            return loss,derivative
        low,high = 0., max(1e-8,min(1e8,info["slope"]/max(info["curvature"],1e-30)))
        best = (start,0.,-info["slope"])
        bracketed,evals = False,0
        for _ in range(self.config["ray_max_expansions"]):
            loss,derivative = at(high)
            evals += 1
            if math.isfinite(loss) and loss<best[0]:
                best = loss,high,derivative
            if math.isfinite(derivative) and derivative>=0:
                bracketed = True
                break
            if not math.isfinite(loss):
                break
            low,high = high,high*2
            if high>self.config["ray_max_fro"]:
                break
        if bracketed:
            for _ in range(self.config["ray_bisections"]):
                mid = (low+high)/2
                _,derivative = at(mid,need_loss=False)
                evals += 1
                if not math.isfinite(derivative) or derivative>=0:
                    high = mid
                else:
                    low = mid
            alpha = (low+high)/2
            loss,derivative = at(alpha)
            evals += 1
            if math.isfinite(loss) and loss<=start+2e-7:
                best = loss,alpha,derivative
        loss,alpha,derivative = best
        result = W+(alpha/norm)*D
        # Store numerical-ray accuracy separately from training-objective replay.
        return result,dict(**info,eta=alpha/norm,eta_fro=alpha,line_evals=evals,
            bracketed=bracketed,terminal_derivative=derivative,ray_loss=loss,
            relative_bracket_width=(high-low)/max(high,1e-30),ray_gain=start-loss)

    @torch.no_grad()
    def diagnostics(self,W,evaluation,singular=None,same_state=False):
        G = evaluation["G"]
        if singular is None:
            singular = torch.linalg.svdvals(self.coordinates(G))
        s = singular.double()
        fro2 = float(s.square().sum())
        info = dict(grad_fro=math.sqrt(fro2),weight_fro=float(torch.linalg.vector_norm(W)))
        if fro2>0:
            info.update(grad_stable_rank=fro2/float(s[0]**2),
                        grad_nuclear_rank=float(s.sum()**2)/fro2,
                        retained_rank=int((s>self.config["polar_rtol"]*s[0]).sum()),
                        grad_top1_energy=float(s[0]**2)/fro2,
                        grad_top32_energy=float(s[:32].square().sum())/fro2,
                        grad_top128_energy=float(s[:128].square().sum())/fro2)
        q = evaluation["q"]
        marginal = q@self.pi
        info["prediction_marginal_tv"] = float(abs(marginal-self.p).sum()/2)
        info["prediction_entropy"] = float(self.pi64@(-(q*q.clamp_min(1e-30).log()).sum(0)).double())
        gm = torch.outer(self.U.T@(marginal-self.p),self.mean_E)
        info["marginal_gradient_norm_ratio"] = float(torch.linalg.vector_norm(gm))/max(math.sqrt(fro2),1e-30)
        if self.heldout is not None:
            rows,cols,weights,_ = self.heldout
            logz=torch.logsumexp(evaluation["z"],0)
            info["heldout_ce"]=float(weights@(logz[cols]-evaluation["z"][rows,cols]).double())
        for name,ids,mass in zip(["head","middle","tail"],self.buckets,self.bucket_mass):
            info[f"ce_{name}"] = float(self.pi64[ids]@evaluation["ce"][ids])/max(mass,1e-30)
            info[f"mass_{name}"] = mass
        probes = []
        if same_state:
            directions = {}
            for method in ["gd_ls","muon_ls","h0_ls","signgd_ls"]:
                directions[method] = self.direction(G,method)[0]
            if self.case.get("diagnostics_preconditioners"):
                for kind in ["input","output","diagonal","wrong_basis","reference_zero","reference_unigram"]:
                    directions[kind] = self.preconditioned(G,kind)
            for method,D in directions.items():
                _,probe = self.ray(W,evaluation,D)
                probes.append(dict(candidate=method,**probe))
        return info,s.cpu().numpy().astype(np.float32),probes
