# --------------------------------------------------------------------------
# GPU capacity: ECS-optimized GPU AMI in an ASG managed by a capacity provider.
# --------------------------------------------------------------------------
data "aws_ssm_parameter" "gpu_ami" {
  name = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

resource "aws_ecs_cluster" "this" {
  name = local.prefix
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = local.tags
}

resource "aws_iam_role" "node" {
  name_prefix        = "hei-node-"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
  tags               = local.tags
}

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy_attachment" "node_ecs" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "node_ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "node" {
  name_prefix = "hei-node-"
  role        = aws_iam_role.node.name
}

resource "aws_launch_template" "gpu" {
  name_prefix   = "${local.prefix}-gpu-"
  image_id      = data.aws_ssm_parameter.gpu_ami.value
  instance_type = var.instance_type

  iam_instance_profile { arn = aws_iam_instance_profile.node.arn }
  vpc_security_group_ids = [aws_security_group.node.id]

  metadata_options {
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 200 # model weights cache
      volume_type = "gp3"
      encrypted   = true
    }
  }

  user_data = base64encode(<<-EOT
    #!/bin/bash
    cat >> /etc/ecs/ecs.config <<EOF
    ECS_CLUSTER=${aws_ecs_cluster.this.name}
    ECS_ENABLE_GPU_SUPPORT=true
    ECS_ENABLE_TASK_IAM_ROLE=true
    ECS_ENABLE_AWSLOGS_EXECUTIONROLE_OVERRIDE=true
    ECS_CONTAINER_START_TIMEOUT=15m
    ECS_IMAGE_PULL_BEHAVIOR=prefer-cached
    EOF
    mkdir -p /opt/hf-cache && chmod 777 /opt/hf-cache
  EOT
  )

  tag_specifications {
    resource_type = "instance"
    tags          = merge(local.tags, { Name = "${local.prefix}-gpu" })
  }
  tags = local.tags
}

resource "aws_autoscaling_group" "gpu" {
  name_prefix           = "${local.prefix}-gpu-"
  min_size              = var.min_nodes
  max_size              = var.max_nodes
  vpc_zone_identifier   = aws_subnet.private[*].id
  protect_from_scale_in = true

  launch_template {
    id      = aws_launch_template.gpu.id
    version = "$Latest"
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }

  lifecycle { ignore_changes = [desired_capacity] }
}

resource "aws_ecs_capacity_provider" "gpu" {
  name = "${local.prefix}-gpu"
  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.gpu.arn
    managed_termination_protection = "ENABLED"
    managed_scaling {
      status          = "ENABLED"
      target_capacity = 100
    }
  }
  tags = local.tags
}

resource "aws_ecs_cluster_capacity_providers" "this" {
  cluster_name       = aws_ecs_cluster.this.name
  capacity_providers = [aws_ecs_capacity_provider.gpu.name]
  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }
}

# --------------------------------------------------------------------------
# IAM for tasks
# --------------------------------------------------------------------------
data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

locals {
  secret_arns = compact([var.anthropic_secret_arn, var.hf_token_secret_arn, var.gateway_api_keys_secret_arn])
}

resource "aws_iam_role" "execution" {
  name_prefix        = "hei-exec-"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_secrets" {
  count = length(local.secret_arns) > 0 ? 1 : 0
  role  = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = local.secret_arns
    }]
  })
}

resource "aws_iam_role" "task" {
  name_prefix        = "hei-task-"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
  tags               = local.tags
}

resource "aws_iam_role_policy" "task" {
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat(
      [{
        Effect   = "Allow"
        Action   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
        Resource = [aws_kinesis_firehose_delivery_stream.records.arn]
      }],
      var.cloud_provider == "bedrock" ? [{
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resource = ["*"]
      }] : []
    )
  })
}

# --------------------------------------------------------------------------
# Task: vLLM (open-weight model) + gateway (gate + escalation) sharing localhost
# --------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "this" {
  name              = "/hybrid-edge/${local.prefix}"
  retention_in_days = 30
  tags              = local.tags
}

locals {
  gateway_secrets = concat(
    var.anthropic_secret_arn == "" ? [] : [{ name = "ANTHROPIC_API_KEY", valueFrom = var.anthropic_secret_arn }],
    var.gateway_api_keys_secret_arn == "" ? [] : [{ name = "HEI_API_KEYS", valueFrom = var.gateway_api_keys_secret_arn }],
  )
  vllm_secrets = var.hf_token_secret_arn == "" ? [] : [{ name = "HF_TOKEN", valueFrom = var.hf_token_secret_arn }]

  logs = {
    logDriver = "awslogs"
    options = {
      "awslogs-group"         = aws_cloudwatch_log_group.this.name
      "awslogs-region"        = local.region
      "awslogs-stream-prefix" = "edge"
    }
  }
}

resource "aws_ecs_task_definition" "edge" {
  family                   = local.prefix
  requires_compatibilities = ["EC2"]
  network_mode             = "awsvpc"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  volume {
    name      = "hf-cache"
    host_path = "/opt/hf-cache"
  }

  container_definitions = jsonencode([
    {
      name      = "vllm"
      image     = var.vllm_image
      essential = true
      command = [
        "--model", var.edge_model,
        "--max-model-len", tostring(var.edge_max_model_len),
        "--gpu-memory-utilization", "0.90",
        "--max-logprobs", "20",
        "--port", "8000",
      ]
      resourceRequirements = [{ type = "GPU", value = "1" }]
      memoryReservation    = 12288
      linuxParameters      = { sharedMemorySize = 4096 }
      environment          = [{ name = "HF_HOME", value = "/hf-cache" }]
      secrets              = local.vllm_secrets
      mountPoints          = [{ sourceVolume = "hf-cache", containerPath = "/hf-cache" }]
      healthCheck = {
        command     = ["CMD-SHELL", "python3 -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)\""]
        interval    = 30
        timeout     = 5
        retries     = 5
        startPeriod = 300
      }
      logConfiguration = local.logs
    },
    {
      name              = "gateway"
      image             = var.gateway_image
      essential         = true
      stopTimeout       = 120 # SIGTERM -> SIGKILL grace; covers uvicorn's 95 s graceful shutdown
      memoryReservation = 1024
      portMappings      = [{ containerPort = 8080, protocol = "tcp" }]
      dependsOn         = [{ containerName = "vllm", condition = "HEALTHY" }]
      environment = [
        { name = "HEI_REGION", value = local.region },
        { name = "HEI_EDGE_ID", value = local.prefix },
        { name = "HEI_EDGE_BACKEND", value = "openai" },
        { name = "HEI_EDGE_BASE_URL", value = "http://127.0.0.1:8000/v1" },
        { name = "HEI_EDGE_MODEL", value = var.edge_model },
        { name = "HEI_EDGE_SYSTEM_ROLE", value = var.edge_system_role },
        { name = "HEI_EDGE_CONTEXT_TOKENS", value = tostring(var.edge_max_model_len) },
        { name = "HEI_GATE_THRESHOLD", value = tostring(var.gate_threshold) },
        { name = "HEI_GATE_CONSISTENCY_SAMPLES", value = tostring(var.gate_consistency_samples) },
        { name = "HEI_CLOUD_PROVIDER", value = var.cloud_provider },
        { name = "HEI_CLOUD_MODEL", value = var.cloud_model },
        { name = "HEI_CLOUD_EFFORT", value = var.cloud_effort },
        { name = "HEI_CLOUD_AWS_REGION", value = var.cloud_aws_region != "" ? var.cloud_aws_region : local.region },
        { name = "HEI_CLOUD_MAX_ESCALATIONS_PER_MIN", value = tostring(var.max_escalations_per_min) },
        { name = "HEI_CLOUD_MAX_ESCALATIONS_PER_KEY_PER_MIN", value = tostring(var.max_escalations_per_key_per_min) },
        { name = "HEI_SHADOW_RATE", value = tostring(var.shadow_rate) },
        { name = "HEI_RECORD_SINK", value = "firehose" },
        { name = "HEI_RECORD_FIREHOSE_STREAM", value = aws_kinesis_firehose_delivery_stream.records.name },
        { name = "HEI_RECORD_INCLUDE_TEXT", value = tostring(var.record_include_text) },
        { name = "HEI_EMIT_EMF", value = "true" },
        # Behind the ALB, which appends the real client address (Global Accelerator preserves it).
        { name = "HEI_TRUST_FORWARDED_FOR", value = "true" },
        { name = "HEI_CLOUD_DEADLINE_S", value = "90" },
      ]
      secrets          = local.gateway_secrets
      logConfiguration = local.logs
    },
  ])

  tags = local.tags
}

resource "aws_ecs_service" "edge" {
  name                              = "edge"
  cluster                           = aws_ecs_cluster.this.id
  task_definition                   = aws_ecs_task_definition.edge.arn
  desired_count                     = var.desired_tasks
  health_check_grace_period_seconds = 900 # first boot downloads model weights
  enable_execute_command            = true

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.gpu.name
    weight            = 1
  }

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.task.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.gateway.arn
    container_name   = "gateway"
    container_port   = 8080
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  ordered_placement_strategy {
    type  = "spread"
    field = "attribute:ecs.availability-zone"
  }

  lifecycle { ignore_changes = [desired_count] }
  depends_on = [aws_lb_listener.http, aws_lb_listener.https, aws_ecs_cluster_capacity_providers.this]
  tags       = local.tags
}

# Scale on request load per task; the capacity provider adds GPU nodes as needed.
resource "aws_appautoscaling_target" "edge" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.edge.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.desired_tasks
  max_capacity       = var.max_nodes
}

resource "aws_appautoscaling_policy" "requests" {
  name               = "${local.prefix}-requests"
  service_namespace  = aws_appautoscaling_target.edge.service_namespace
  resource_id        = aws_appautoscaling_target.edge.resource_id
  scalable_dimension = aws_appautoscaling_target.edge.scalable_dimension
  policy_type        = "TargetTrackingScaling"

  target_tracking_scaling_policy_configuration {
    target_value       = 300 # requests per target per minute
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "ALBRequestCountPerTarget"
      resource_label         = "${aws_lb.this.arn_suffix}/${aws_lb_target_group.gateway.arn_suffix}"
    }
  }
}
