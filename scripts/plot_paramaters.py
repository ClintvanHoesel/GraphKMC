# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:45:52 2022

@author: s164097
"""

import matplotlib.pyplot as plt

cm_to_inch = 0.393701
inch_to_cm = 1.0 / cm_to_inch


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def set_plot_parameters(
    update_dict=None,
    nts=15.0,
    bts=18.0,
    form="jpeg",
    w=12.0,
    h=9.0,
    mats=10.0,
    mits=7.0,
    axlw=2.0,
    matlw=1.5,
    mitlw=1.5,
):
    # NORMAL_SIZE = 18.
    # BIGGER_SIZE = 22.
    # form = "svg"
    plt.rcParams.update(
        {
            "ps.usedistiller": "xpdf",
            "text.latex.preamble": " ".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{stix2}",
                    r"\newcommand*\mean[1]{\overline{#1}}",
                    r"\newcommand{\matr}[1]{\matrixsym{#1}}",
                    r"\newcommand{\vect}[1]{\vectorsym{#1}}",
                    r"\newcommand{\tens}[1]{\tensorsym{#1}}",
                    r"\newcommand{\of}[1]{\left(#1\right)}",
                    r"\newcommand{\off}[1]{\left[#1\right]}",
                    r"\newcommand{\offf}[1]{\left\{#1\right\}}",
                    r"\newcommand{\abss}[1]{\left|#1\right|}",
                    r"\newcommand{\innerprod}[1]{\left(#1\right)}",
                    r"\newcommand{\expec}[1]{\left<#1\right>}",
                    r"\newcommand{\tm}[1]{\text{#1}}",
                    r"\newcommand{\ofl}[1]{\left(#1\right.}",
                    r"\newcommand{\ofr}[1]{\left.#1\right)}",
                    r"\newcommand{\ofi}[1]{\left.#1\right.}",
                    r"\newcommand{\matt}[1]{\text{#1}}",
                    r"\newcommand{\funct}[1]{\matt{#1}}",
                    r"\newcommand{\diff}{\mathop{}\!\mathrm{d}}",
                    r"\newcommand{\Diff}[1]{\mathop{}\!\mathrm{d^#1}}",
                ]
            ),
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.size": nts,
            "axes.linewidth": axlw,
            "xtick.top": True,
            "xtick.bottom": True,
            "ytick.left": True,
            "ytick.right": True,
            "axes.spines.right": True,
            "axes.spines.left": True,
            "axes.spines.top": True,
            "axes.spines.bottom": True,
            "xtick.major.size": mats,
            "ytick.major.size": mats,
            "xtick.major.width": matlw,
            "ytick.major.width": matlw,
            "xtick.minor.size": mits,
            "ytick.minor.size": mits,
            "xtick.minor.width": mitlw,
            "ytick.minor.width": mitlw,
            "xtick.labelsize": nts,
            "ytick.labelsize": nts,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.format": form,
            "savefig.transparent": True,
            "axes.titlesize": nts,
            "axes.labelsize": nts,
            "xtick.labelsize": nts,
            "ytick.labelsize": nts,
            "legend.fontsize": nts,
            "legend.frameon": True,
            "figure.titlesize": bts,
            "figure.labelsize": nts,
            "figure.figsize": cm2inch((w, h)),
            "figure.constrained_layout.use": True,
            "figure.dpi": 300,
        }
    )

    if update_dict:
        plt.rcParams.update(update_dict)
