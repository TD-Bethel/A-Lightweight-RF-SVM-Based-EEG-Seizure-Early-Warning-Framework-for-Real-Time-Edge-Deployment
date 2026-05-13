# LaTeX Flow Chart Figures

Use these split figures when the full NeuroWatch flow chart is too large for one LaTeX page.

Add these packages/settings in your LaTeX preamble:

```latex
\usepackage{graphicx}
\usepackage{float}

\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{8pt}
\setlength{\@fpbot}{0pt plus 1fil}
\makeatother
```

Then insert the figures like this:

```latex
\clearpage
\begin{figure}[p]
    \centering
    \includegraphics[
        width=0.95\textwidth,
        height=0.82\textheight,
        keepaspectratio
    ]{docs/flowchart_part_1_training.png}
    \caption{NeuroWatch training pipeline and saved model artifacts.}
    \label{fig:neurowatch-training-flow}
\end{figure}

\clearpage
\begin{figure}[p]
    \centering

    \textbf{(a) Live Raspberry Pi monitoring pipeline and hardware outputs}

    \vspace{0.3em}

    \includegraphics[
        width=0.96\textwidth,
        height=0.39\textheight,
        keepaspectratio
    ]{docs/flowchart_part_2_runtime_hardware.png}

    \vspace{1.0em}

    \textbf{(b) Runtime JSON outputs consumed by the dashboard and patient mobile UI}

    \vspace{0.3em}

    \includegraphics[
        width=0.96\textwidth,
        height=0.39\textheight,
        keepaspectratio
    ]{docs/flowchart_part_3_dashboard.png}

    \caption{Live monitoring outputs and dashboard integration.}
    \label{fig:neurowatch-runtime-dashboard-flow}
\end{figure}
\clearpage
```

If the first figure is still too small because it has the most nodes, use `height=0.78\textheight` for that figure, or keep it as two separate figures.
