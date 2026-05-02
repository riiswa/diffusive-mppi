"""
envs.py — environment registry for DNI-MPPI experiments.

Each entry exposes:
  step_fn, cost_fn, obs_fn, obs_label, goal_line, obs_ylim,
  reset_fn, action_clip, state_range,
  N (horizon), max_steps, sigma_u, sigma_0, lf_beta,
  action_cost (bool, dense only)
"""

import jax.numpy as jnp
from jax import random


# ── helpers ───────────────────────────────────────────────────────────────────

def wrap(x):
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


# ── dynamics ──────────────────────────────────────────────────────────────────

def mc_step(s, a):
    pos, vel = s[0], s[1]
    vel = jnp.clip(vel + 0.0015 * jnp.clip(a, -1., 1.) - 0.0025 * jnp.cos(3.*pos), -0.07, 0.07)
    pos = jnp.clip(pos + vel, -1.2, 0.6)
    vel = jnp.where(pos <= -1.2, 0., vel)
    return jnp.stack([pos, vel])


def pendulum_step(s, a):
    g, m, l, dt = 10., 1., 1., 0.05
    th, td = s[0], s[1]
    td = jnp.clip(td + (3.*g/(2.*l)*jnp.sin(th) + 3./(m*l**2)*jnp.clip(a, -2., 2.))*dt, -8., 8.)
    return jnp.stack([wrap(th + td*dt), td])


def cartpole_step(s, a):
    mc, mp, l, g, dt = 1., 0.1, 0.5, 9.8, 0.02
    x, xd, th, td = s[0], s[1], s[2], s[3]
    u = jnp.clip(a, -10., 10.)
    st, ct = jnp.sin(th), jnp.cos(th)
    M, ml  = mc + mp, mp * l
    tmp    = (u + ml*td**2*st) / M
    tdd    = (g*st - ct*tmp) / (l*(4./3. - mp*ct**2/M))
    xdd    = tmp - ml*tdd*ct/M
    return jnp.stack([x+xd*dt, xd+xdd*dt, wrap(th+td*dt), td+tdd*dt])


def acrobot_step(s, a):
    m1,m2,l1,lc1,lc2,I1,I2 = 1.,1.,1.,0.5,0.5,1.,1.
    g, dt = 9.8, 0.2
    u = jnp.clip(a, -1., 1.)
    def deriv(s):
        t1,t2,t1d,t2d = s[0],s[1],s[2],s[3]
        d1   = m1*lc1**2+m2*(l1**2+lc2**2+2*l1*lc2*jnp.cos(t2))+I1+I2
        d2   = m2*(lc2**2+l1*lc2*jnp.cos(t2))+I2
        phi2 = m2*lc2*g*jnp.cos(t1+t2-jnp.pi/2)
        phi1 = (-m2*l1*lc2*t2d**2*jnp.sin(t2) - 2*m2*l1*lc2*t2d*t1d*jnp.sin(t2)
                + (m1*lc1+m2*l1)*g*jnp.cos(t1-jnp.pi/2) + phi2)
        t2dd = (u + d2/d1*phi1 - m2*l1*lc2*t1d**2*jnp.sin(t2) - phi2) \
               / (m2*lc2**2 + I2 - d2**2/d1)
        t1dd = -(d2*t2dd + phi1)/d1
        return jnp.stack([t1d,t2d,t1dd,t2dd])
    k1=deriv(s); k2=deriv(s+dt/2*k1); k3=deriv(s+dt/2*k2); k4=deriv(s+dt*k3)
    ns = s + dt/6*(k1+2*k2+2*k3+k4)
    return jnp.stack([wrap(ns[0]), wrap(ns[1]),
                      jnp.clip(ns[2],-4*jnp.pi,4*jnp.pi),
                      jnp.clip(ns[3],-9*jnp.pi,9*jnp.pi)])


def double_cartpole_step(s, a):
    Mc,m1,m2,L1,L2,g,dt = 1.,0.1,0.1,0.5,0.3,9.8,0.02
    x,xd,t1,t1d,t2,t2d  = (s[i] for i in range(6))
    u = jnp.clip(a, -20., 20.)
    s1,c1 = jnp.sin(t1), jnp.cos(t1)
    s2,c2 = jnp.sin(t2), jnp.cos(t2)
    M00=Mc+m1+m2; M01=m1*L1*c1; M02=m2*L2*c2; M11=m1*L1**2; M22=m2*L2**2
    b0=u+m1*L1*t1d**2*s1+m2*L2*t2d**2*s2; b1=m1*g*L1*s1; b2=m2*g*L2*s2
    C00=M11*M22; C01=-(M01*M22-M02*0.); C02=-M11*M02
    C11=M00*M22-M02**2; C12=M01*M02; C22=M00*M11-M01**2
    det=M00*C00+M01*C01+M02*C02; inv=1./det
    a0=inv*(C00*b0+C01*b1+C02*b2)
    a1=inv*(C01*b0+C11*b1+C12*b2)
    a2=inv*(C02*b0+C12*b1+C22*b2)
    return jnp.stack([x+xd*dt, xd+a0*dt, wrap(t1+t1d*dt),
                      jnp.clip(t1d+a1*dt,-15.,15.),
                      wrap(t2+t2d*dt),
                      jnp.clip(t2d+a2*dt,-15.,15.)])


def furuta_step(s, a):
    m2=0.024; L0,l2=0.085,0.129; J0,J2=5.72e-5,1.33e-4; g,b0,b2=9.81,1e-4,2.8e-4; dt=0.02
    arm,pole,armd,poled = s[0],s[1],s[2],s[3]
    tau = jnp.clip(a, -0.1, 0.1)
    sp,cp = jnp.sin(pole), jnp.cos(pole)
    M11=J0+m2*L0**2+m2*l2**2*sp**2; M12=m2*L0*l2*cp; M22=J2+m2*l2**2; det=M11*M22-M12**2
    f1=tau-b0*armd-m2*l2**2*jnp.sin(2*pole)*poled*armd+m2*L0*l2*sp*poled**2
    f2=-b2*poled+0.5*m2*l2**2*jnp.sin(2*pole)*armd**2-m2*g*l2*sp
    armdd=(M22*f1-M12*f2)/det; poledd=(-M12*f1+M11*f2)/det
    return jnp.stack([wrap(arm+armd*dt), wrap(pole+poled*dt),
                      jnp.clip(armd+armdd*dt,-20.,20.),
                      jnp.clip(poled+poledd*dt,-20.,20.)])


# ── sparse registry ───────────────────────────────────────────────────────────

ENVS = {
    'Mountain Car': dict(
        step_fn=mc_step,
        cost_fn=lambda s: jnp.where(s[0] >= 0.45, 0., 1.),
        obs_fn=lambda s: s[0], obs_label='position',
        goal_line=0.45, obs_ylim=(-1.3, 0.65),
        reset_fn=lambda k: jnp.array([random.uniform(k, shape=(), minval=-0.6, maxval=-0.4), 0.]),
        action_clip=1., state_range=jnp.array([1.8, 0.14]),
        N=200, max_steps=200, sigma_u=0.5, sigma_0=0.01, lf_beta=2., action_cost=False,
    ),
    'Pendulum': dict(
        step_fn=pendulum_step,
        cost_fn=lambda s: jnp.where((jnp.abs(s[0]) < 0.2) & (jnp.abs(s[1]) < 1.), 0., 1.),
        obs_fn=lambda s: jnp.cos(s[0]), obs_label='cos θ',
        goal_line=float(jnp.cos(jnp.array(0.2))), obs_ylim=(-1.1, 1.1),
        reset_fn=lambda k: jnp.array([wrap(jnp.pi + random.uniform(k, shape=(), minval=-0.1, maxval=0.1)),
                                       random.uniform(k, shape=(), minval=-0.1, maxval=0.1)]),
        action_clip=2., state_range=jnp.array([2*jnp.pi, 16.]),
        N=200, max_steps=500, sigma_u=0.5, sigma_0=0.01, lf_beta=2., action_cost=False,
    ),
    'Cartpole': dict(
        step_fn=cartpole_step,
        cost_fn=lambda s: jnp.where((jnp.cos(s[2]) > 0.95) & (jnp.abs(s[0]) < 2.4), 0., 1.),
        obs_fn=lambda s: jnp.cos(s[2]), obs_label='cos θ',
        goal_line=0.95, obs_ylim=(-1.1, 1.1),
        reset_fn=lambda k: jnp.array([random.uniform(k, shape=(), minval=-0.05, maxval=0.05), 0.,
                                       wrap(jnp.pi + random.uniform(k, shape=(), minval=-0.05, maxval=0.05)), 0.]),
        action_clip=10., state_range=jnp.array([4.8, 10., 2*jnp.pi, 20.]),
        N=200, max_steps=500, sigma_u=1., sigma_0=0.01, lf_beta=2., action_cost=False,
    ),
    # 'Double Cartpole': dict(
    #     step_fn=double_cartpole_step,
    #     cost_fn=lambda s: jnp.where((jnp.cos(s[2])>0.95)&(jnp.cos(s[4])>0.95)&(jnp.abs(s[0])<2.4),0.,1.),
    #     obs_fn=lambda s: (jnp.cos(s[2])+jnp.cos(s[4]))/2., obs_label='mean cos θ',
    #     goal_line=0.95, obs_ylim=(-1.1, 1.1),
    #     reset_fn=lambda k: jnp.array([random.uniform(k, shape=(), minval=-0.05, maxval=0.05), 0.,
    #                                    wrap(jnp.pi+random.uniform(k, shape=(), minval=-0.05, maxval=0.05)), 0.,
    #                                    wrap(jnp.pi+random.uniform(k, shape=(), minval=-0.05, maxval=0.05)), 0.]),
    #     action_clip=20., state_range=jnp.array([4.8, 15., 2*jnp.pi, 20., 2*jnp.pi, 20.]),
    #     N=200, max_steps=500, sigma_u=2., sigma_0=0.01, lf_beta=2., action_cost=False,
    # ),
    'Acrobot': dict(
        step_fn=acrobot_step,
        cost_fn=lambda s: jnp.where(-jnp.cos(s[0])-jnp.cos(s[0]+s[1]) >= 1., 0., 1.),
        obs_fn=lambda s: -jnp.cos(s[0])-jnp.cos(s[0]+s[1]), obs_label='tip height',
        goal_line=1., obs_ylim=(-2.1, 2.1),
        reset_fn=lambda k: jnp.array([random.uniform(k, shape=(), minval=-0.1, maxval=0.1),
                                       random.uniform(k, shape=(), minval=-0.1, maxval=0.1),
                                       random.uniform(k, shape=(), minval=-0.1, maxval=0.1),
                                       random.uniform(k, shape=(), minval=-0.1, maxval=0.1)]),
        action_clip=1., state_range=jnp.array([2*jnp.pi, 2*jnp.pi, 8*jnp.pi, 18*jnp.pi]),
        N=200, max_steps=500, sigma_u=0.5, sigma_0=0.01, lf_beta=2., action_cost=False,
    ),
    'Furuta': dict(
        step_fn=furuta_step,
        cost_fn=lambda s: jnp.where(jnp.abs(s[1]) < 0.2, 0., 1.),
        obs_fn=lambda s: jnp.cos(s[1]), obs_label='cos(pole)',
        goal_line=float(jnp.cos(jnp.array(0.2))), obs_ylim=(-1.1, 1.1),
        reset_fn=lambda k: jnp.array([random.uniform(k, shape=(), minval=-jnp.pi, maxval=jnp.pi),
                                       wrap(jnp.pi+random.uniform(k, shape=(), minval=-0.1, maxval=0.1)),
                                       0., 0.]),
        action_clip=0.1, state_range=jnp.array([2*jnp.pi, 2*jnp.pi, 20., 20.]),
        N=200, max_steps=500, sigma_u=0.02, sigma_0=0.01, lf_beta=2., action_cost=False,
    ),
}


# ── dense registry ────────────────────────────────────────────────────────────

ENVS_DENSE = {
    'Mountain Car': dict(
        **{k: v for k, v in ENVS['Mountain Car'].items()
           if k not in ('cost_fn', 'action_cost', 'max_steps')},
        cost_fn=lambda s: jnp.clip((0.45-s[0])/1.65, 0., 1.),
        max_steps=200, action_cost=True,
    ),
    'Pendulum': dict(
        **{k: v for k, v in ENVS['Pendulum'].items()
           if k not in ('cost_fn', 'action_cost')},
        cost_fn=lambda s: jnp.clip((s[0]**2+0.1*s[1]**2)/(jnp.pi**2+6.4), 0., 1.),
        action_cost=True,
    ),
    'Cartpole': dict(
        **{k: v for k, v in ENVS['Cartpole'].items()
           if k not in ('cost_fn', 'action_cost')},
        cost_fn=lambda s: jnp.clip((1.-jnp.cos(s[2]))/2.+0.1*(s[0]/2.4)**2, 0., 1.),
        action_cost=True,
    ),
    # 'Double Cartpole': dict(
    #     **{k: v for k, v in ENVS['Double Cartpole'].items()
    #        if k not in ('cost_fn', 'action_cost')},
    #     cost_fn=lambda s: jnp.clip(
    #         (1.-jnp.cos(s[2]))/2.*0.45+(1.-jnp.cos(s[4]))/2.*0.45+0.1*(s[0]/2.4)**2, 0., 1.),
    #     action_cost=True,
    # ),
    'Acrobot': dict(
        **{k: v for k, v in ENVS['Acrobot'].items()
           if k not in ('cost_fn', 'action_cost')},
        cost_fn=lambda s: jnp.clip((1.-(-jnp.cos(s[0])-jnp.cos(s[0]+s[1])))/3., 0., 1.),
        action_cost=True,
    ),
    'Furuta': dict(
        **{k: v for k, v in ENVS['Furuta'].items()
           if k not in ('cost_fn', 'action_cost')},
        cost_fn=lambda s: jnp.clip((1.-jnp.cos(s[1]))/2.+0.05*(s[2]/20.)**2, 0., 1.),
        action_cost=True,
    ),
}