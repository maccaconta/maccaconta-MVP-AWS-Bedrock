variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "aws_account_id" {
  type = string
}

variable "cluster_name" {
  type    = string
  default = "ems-mvp-cluster"
}

variable "node_group_name" {
  type    = string
  default = "standard-workers-small"
}

variable "node_instance_type" {
  type    = string
  default = "t3.small"
}

variable "node_desired_capacity" {
  type    = number
  default = 1
}

variable "ecr_repository" {
  type    = string
  default = "ems-mvp"
}

variable "image_tag" {
  type    = string
  default = "latest"
}

# Service account/IRSA
variable "k8s_namespace" {
  type    = string
  default = "default"
}

variable "k8s_service_account" {
  type    = string
  default = "ems-mvp-sa"
}

variable "bedrock_kb_id" {
  type = string
  default = ""
}