########################
# ECR repository
########################
resource "aws_ecr_repository" "app" {
  name = var.ecr_repository

  image_scanning_configuration {
    scan_on_push = false
  }

  encryption_configuration {
    encryption_type = "AES256"
  }
}

########################
# EKS cluster (uses managed node group)
# We use the official terraform-aws-modules/eks/aws module
########################
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "22.0.0"

  cluster_name    = var.cluster_name
  cluster_version = "1.34"
  subnets         = null
  vpc_id          = null
  manage_aws_auth = true

  # Create a VPC for convenience. In corporate infra you will attach to existing VPC.
  create_vpc = true
  vpc_cidr = "10.100.0.0/16"
  private_subnets = ["10.100.1.0/24", "10.100.2.0/24", "10.100.3.0/24"]
  public_subnets  = ["10.100.101.0/24", "10.100.102.0/24"]

  node_groups = {
    "${var.node_group_name}" = {
      desired_capacity = var.node_desired_capacity
      max_capacity     = var.node_desired_capacity
      min_capacity     = var.node_desired_capacity
      instance_types   = [var.node_instance_type]
      ami_type         = "AL2_x86_64"
    }
  }

  tags = {
    Project = "ems-mvp"
    Env     = "managed-by-terraform"
  }
}

########################
# IAM Policy for Bedrock
########################
data "aws_iam_policy_document" "bedrock" {
  statement {
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:Retrieve",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_policy" "bedrock_policy" {
  name   = "BedrockAccessPolicy"
  policy = data.aws_iam_policy_document.bedrock.json
}

########################
# Create IAM role for the Kubernetes ServiceAccount (IRSA)
# Using module's iam_role_with_oidc feature via eks module outputs
########################
resource "aws_iam_role" "sa_role" {
  name = "${var.cluster_name}-${var.k8s_service_account}-role"

  assume_role_policy = data.aws_iam_policy_document.sa_assume_role.json
}

data "aws_iam_policy_document" "sa_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer, "https://", "")}:sub"
      values   = ["system:serviceaccount:${var.k8s_namespace}:${var.k8s_service_account}"]
    }

    actions = ["sts:AssumeRoleWithWebIdentity"]
  }
}

resource "aws_iam_role_policy_attachment" "attach_bedrock" {
  role       = aws_iam_role.sa_role.name
  policy_arn = aws_iam_policy.bedrock_policy.arn
}

########################
# Kubernetes: create ServiceAccount with annotation for IRSA
# This uses the kubernetes provider implicitly via local-exec and kubectl.
# For simplicity we provide a kubectl apply manifest generated afterwards.
########################
resource "local_file" "sa_manifest" {
  content = <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${var.k8s_service_account}
  namespace: ${var.k8s_namespace}
  annotations:
    eks.amazonaws.com/role-arn: ${aws_iam_role.sa_role.arn}
EOF
  filename = "${path.module}/k8s_sa.yaml"
}

resource "null_resource" "apply_service_account" {
  provisioner "local-exec" {
    command = "kubectl apply -f ${local_file.sa_manifest.filename} --context ${module.eks.kubeconfig_context}"
    environment = {
      KUBECONFIG = module.eks.kubeconfig_filename
    }
  }
  depends_on = [module.eks, local_file.sa_manifest]
}

########################
# Outputs
########################
output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "cluster_name" {
  value = module.eks.cluster_id
}

output "kubeconfig" {
  value = module.eks.kubeconfig_filename
}