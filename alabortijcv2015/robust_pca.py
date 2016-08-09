from __future__ import division
import scipy.linalg as la
from menpo.visualize import print_dynamic
import numpy as np


def svd(X, k=-1):
    U, S, V = la.svd(X, full_matrices=False)
    if k < 0:
        return U, S, V
    else:
        return U[:, :k], S[:k], V[:k, :]


def _verbose(A, E, D):
    A_rank = np.linalg.matrix_rank(A)
    perc_E = (np.count_nonzero(E) / E.size) * 100
    error = la.norm(D - A - E, ord='fro')
    print_dynamic('rank(A): {}, |E|_1: {:.2f}%, |D-A-E|_F: {:.2e}'.format(
        A_rank, perc_E, error))


def rpca_alm(X, lmbda=None, tol=1e-7, max_iters=1000, verbose=True,
             inexact=True):
    """
    Augmented Lagrange Multiplier
    """
    if lmbda is None:
        lmbda = 1.0 / np.sqrt(X.shape[0])

    Y = np.sign(X)
    norm_two = svd(Y, 1)[1]
    norm_inf = np.abs(Y).max() / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm

    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)

    dnorm = la.norm(X, ord='fro')
    tol_primal = 1e-6 * dnorm
    total_svd = 0
    mu = 0.5 / norm_two
    rho = 6

    sv = 5
    n = Y.shape[0]

    for iter1 in range(max_iters):
        primal_converged = False
        sv = sv + np.round(n * 0.1)
        primal_iter = 0

        while not primal_converged:
            Eraw = X - A + (1/mu) * Y
            Eupdate = np.maximum(
                Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
            U, S, V = svd(X - Eupdate + (1 / mu) * Y, sv)

            svp = (S > 1/mu).sum()
            if svp < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(.05 * n), n])

            Aupdate = np.dot(
                np.dot(U[:, :svp], np.diag(S[:svp] - 1/mu)), V[:svp, :])

            if primal_iter % 10 == 0 and verbose >= 2:
                print(la.norm(A - Aupdate, ord='fro'))

            if ((la.norm(A - Aupdate, ord='fro') < tol_primal and
                la.norm(E - Eupdate, ord='fro') < tol_primal) or
                (inexact and primal_iter > 5)):
                primal_converged = True

            A = Aupdate
            E = Eupdate
            primal_iter += 1
            total_svd += 1

        Z = X - A - E
        Y = Y + mu * Z
        mu *= rho

        if la.norm(Z, ord='fro') / dnorm < tol:
            if verbose:
                print('\nConverged at iteration {}'.format(iter1))
            break

        if verbose:
            _verbose(A, E, X)

    return A, E


def rpca_alm_no_E(X, tol=1e-7, max_iters=1000, verbose=True,
                  inexact=True):
    """
    Augmented Lagrange Multiplier
    """
    Y = np.sign(X)
    norm_two = svd(Y, 1)[1]
    Y = Y / norm_two

    A = np.zeros(Y.shape)

    dnorm = la.norm(X, ord='fro')
    tol_primal = 1e-6 * dnorm
    total_svd = 0
    mu = 0.5 / norm_two
    rho = 6

    sv = 5
    n = Y.shape[0]

    for iter1 in range(max_iters):
        primal_converged = False
        sv = sv + np.round(n * 0.1)
        primal_iter = 0

        while not primal_converged:
            U, S, V = svd(X - (1 / mu) * Y, sv)

            svp = (S > 1/mu).sum()
            if svp < sv:
                sv = np.min([svp + 1, n])
            else:
                sv = np.min([svp + round(.05 * n), n])

            Aupdate = np.dot(
                np.dot(U[:, :svp], np.diag(S[:svp] - 1/mu)), V[:svp, :])

            if primal_iter % 10 == 0 and verbose >= 2:
                print(la.norm(A - Aupdate, ord='fro'))

            if (la.norm(A - Aupdate, ord='fro') < tol_primal or
                    (inexact and primal_iter > 5)):
                primal_converged = True

            A = Aupdate
            primal_iter += 1
            total_svd += 1

        Z = X - A
        Y = Y + mu * Z
        mu *= rho

        if la.norm(Z, ord='fro') / dnorm < tol:
            if verbose:
                print('\nConverged at iteration {}'.format(iter1))
            break

        if verbose:
            _verbose(A, None, X)

    return A


def rpca_pcp(X, lamda=None, max_iters=1000, tol=1.0e-7, verbose=True):
    m, n = X.shape
    # Set params
    if lamda is None:
        lamda = 1.0 / np.sqrt(min(m, n))
    # Initialize
    Y = X
    u, s, v = svd(Y, k=1)
    norm_two = s[0]
    norm_inf = la.norm(Y.ravel(), ord=np.inf) / lamda
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    A_hat = np.zeros((m, n))
    mu = 1.25/norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = np.linalg.norm(X, 'fro')

    num_iters = 0
    total_svd = 0
    sv = 10
    while True:
        num_iters += 1

        temp_T = X - A_hat + (1/mu)*Y
        E_hat = np.maximum(temp_T - lamda/mu, 0)
        E_hat = E_hat + np.minimum(temp_T + lamda/mu, 0)

        u, s, v = svd(X - E_hat + (1/mu)*Y, k=sv)
        svp = np.sum(s > 1/mu)

        if svp < sv:
            sv = min(svp + 1, n)
        else:
            sv = min(svp + round(0.05*n), n)

        A_hat = np.dot(
            np.dot(
                u[:, :svp],
                np.diag(s[:svp] - 1 / mu)
            ),
            v[:svp, :]
        )

        total_svd += 1

        Z = X - A_hat - E_hat

        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if verbose:
            _verbose(A_hat, E_hat, X)

        if (la.norm(Z, ord='fro') / d_norm < tol) or num_iters >= max_iters:
            return A_hat, E_hat


def explicit_rank_pcp(X, k, lamda=None, max_iters=1000, tol=1.0e-7, verbose=True):
    m, n = X.shape
    # Set params
    if lamda is None:
        lamda = 1.0 / np.sqrt(min(m, n))

    # Initialize
    Y = X
    u, s, v = svd(Y, k=1)
    norm_two = s[0]
    norm_inf = la.norm(Y.ravel(), ord=np.inf) / lamda
    dual_norm = max(norm_two, norm_inf)
    Y = Y / dual_norm

    E = np.zeros((m, n))
    J = np.zeros((k, n))

    mu = 1.25/norm_two
    mu_bar = mu * 1e7
    rho = 1.1
    d_norm = np.linalg.norm(X, 'fro')

    num_iters = 0

    while True:
        num_iters += 1

        # Q
        dey = X - E + Y / mu
        temp = dey.dot(J.T)
        U, sigma, V = svd(temp)
        Q = U.dot(V.T)

        # J
        temp = Q.T.dot(dey)

        U, sigma, V = svd(temp)
        svp = np.sum(sigma > 1/mu)
        if svp >= 1:
            sigma = sigma[:svp] - 1.0 / mu
        else:
            svp = 1
            sigma = [0]

        # J = U * S * V'
        J = U[:, :svp].dot(
                np.diag(sigma).dot(V[:svp, :]))

        # E
        A = Q.dot(J)
        temp = X - A + Y / mu
        E = np.maximum(temp - lamda / mu, 0)
        E = E + np.minimum(temp + lamda / mu, 0)

        Z = X - A - E
        Y = Y + mu * Z
        mu = min(mu * rho, mu_bar)

        if verbose:
            _verbose(A, E, X)

        if (la.norm(Z, ord='fro') / d_norm < tol) or num_iters >= max_iters:
            return A, E
