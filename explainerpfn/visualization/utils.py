import torch


def get_pdfs(full_outputs):
    """
    Get the pdfs from the full prediction output of ExplainerPFN.
    The output is given as a tuple of (pdfs, cdfs, bucket_means).
    """
    criterion = full_outputs["criterion"]
    logits = full_outputs["logits"]

    bucket_means = criterion.borders[:-1] + criterion.bucket_widths / 2
    pdf_probs = torch.softmax(logits, -1)

    side_normals = (
        criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[0]),
        criterion.halfnormal_with_p_weight_before(criterion.bucket_widths[-1]),
    )
    bucket_means[0] = -side_normals[0].mean + criterion.borders[1]
    bucket_means[-1] = side_normals[1].mean + criterion.borders[-2]

    cdf_probs = torch.cumsum(pdf_probs, dim=-1)
    return pdf_probs, cdf_probs, bucket_means
